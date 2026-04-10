import random
import json
import logging
from collections import Counter, defaultdict
import pandas as pd
import torch
from rdkit.Chem import AllChem
from copy import deepcopy
from tqdm import trange
from rdkit import Chem
from rdkit.rdBase import DisableLog
import argparse
import os
from model.Molecule_representation.datasets.bace_geomol_feat import featurize_mol_from_smiles
from icecream import install
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import seaborn

import yaml
from model.Molecule_representation.datasets.custom_collate import *  # do not remove
from model.Molecule_representation.models import *  # do not remove
from torch.nn import *  # do not remove
from torch.optim import *  # do not remove
from model.Molecule_representation.commons.losses import *  # do not remove
from torch.optim.lr_scheduler import *  # do not remove
from model.Molecule_representation.datasets.samplers import *  # do not remove

from torch_geometric.utils import to_dgl
# turn on for debugging C code like Segmentation Faults
import faulthandler
from transformers import AutoModel,AutoTokenizer,T5ForConditionalGeneration
faulthandler.enable()
install()
seaborn.set_theme()
DisableLog('rdApp.warning')
DisableLog('rdApp.error')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FUSION_CHECKPOINT_PATH = os.path.join(BASE_DIR, 'model', 'value_function_text_conditioned_fusion-model_val.pkl')
reactant_tokenizer = None
reactant_model = None

class ValueMLP(nn.Module):
    def __init__(self, n_layers, fp_dim, latent_dim, dropout_rate):
        super(ValueMLP, self).__init__()
        self.n_layers = n_layers
        self.fp_dim = fp_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        logging.info('Initializing value model: latent_dim=%d' % self.latent_dim)

        layers = []
        layers.append(nn.Linear(fp_dim, latent_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))
        for _ in range(self.n_layers - 1):
            layers.append(nn.Linear(latent_dim, latent_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
        layers.append(nn.Linear(latent_dim, 1))

        self.layers = nn.Sequential(*layers)

    def forward(self, fps):
        x = fps
        x = self.layers(x)
        x = torch.log(1 + torch.exp(x))

        return x

class TextConditionedFusionModel(nn.Module):
    def __init__(self, d_molecule, d_text, d_model, dropout_rate):

        super(TextConditionedFusionModel, self).__init__()
        self.d_molecule = d_molecule
        self.d_text = d_text
        self.d_model = d_model

        self.molecule_proj = nn.Sequential(
            nn.Linear(d_molecule, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        self.text_proj = nn.Sequential(
            nn.Linear(d_text, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        self.gamma = nn.Linear(d_model, d_model)
        self.beta = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.output_proj = nn.Sequential(
            nn.Linear(d_model * 5, d_model),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, d_model)
        )

    def _prepare_inputs(self, molecule_embedding, text_embedding):
        if molecule_embedding.dim() == 1:
            molecule_embedding = molecule_embedding.unsqueeze(0)
        if text_embedding.dim() == 1:
            text_embedding = text_embedding.unsqueeze(0)

        if molecule_embedding.dim() == 3 and molecule_embedding.size(1) == 1:
            molecule_embedding = molecule_embedding.squeeze(1)
        if text_embedding.dim() == 3 and text_embedding.size(1) == 1:
            text_embedding = text_embedding.squeeze(1)

        if text_embedding.size(0) == 1 and molecule_embedding.size(0) > 1:
            text_embedding = text_embedding.expand(molecule_embedding.size(0), -1)
        elif molecule_embedding.size(0) != text_embedding.size(0):
            raise ValueError(
                f"Expected molecule and text batches to align, got {molecule_embedding.shape} and {text_embedding.shape}"
            )

        return molecule_embedding, text_embedding

    def forward(self, molecule_embedding, text_embedding):
        molecule_embedding, text_embedding = self._prepare_inputs(molecule_embedding, text_embedding)

        molecule_feature = self.molecule_proj(molecule_embedding)
        text_feature = self.text_proj(text_embedding)

        gamma = torch.tanh(self.gamma(text_feature))
        beta = self.beta(text_feature)
        modulated_molecule = molecule_feature * (1 + gamma) + beta

        gate = self.gate(torch.cat([molecule_feature, text_feature], dim=-1))
        fused_core = gate * modulated_molecule + (1 - gate) * molecule_feature

        interaction_feature = molecule_feature * text_feature
        difference_feature = torch.abs(molecule_feature - text_feature)
        fusion_feature = torch.cat(
            [molecule_feature, fused_core, text_feature, interaction_feature, difference_feature],
            dim=-1
        )
        return self.output_proj(fusion_feature)

class FusionModel(nn.Module):
    def __init__(self, d_molecule, d_text, d_model, n_layers, latent_dim, dropout_rate):
        super(FusionModel, self).__init__()
        self.text_conditioned_fusion = TextConditionedFusionModel(d_molecule, d_text, d_model, dropout_rate)
        self.value_mlp = ValueMLP(n_layers, d_model, latent_dim, dropout_rate)

    def forward(self, molecule_embedding, text_embedding):
        fused_output = self.text_conditioned_fusion(molecule_embedding, text_embedding)
        value_output = self.value_mlp(fused_output)
        return value_output.sum()

def smiles_to_fp(s, fp_dim=600, pack=False):
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {s}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits(), dtype=np.bool_)
    arr[onbits] = 1

    if pack:
        arr = np.packbits(arr)
    fp = 1 * np.array(arr)

    return fp


def smiles_to_inchikey_prefix(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToInchiKey(mol)[:14]

def value_fn(smi, text_embedding_tensor):
    with torch.no_grad():
        return _value_fn_impl(smi, text_embedding_tensor)


def _value_fn_impl(smi, text_embedding_tensor):
    if '.' in smi:
        represent = []
        for s in smi.split('.'):
            try:
                g = featurize_mol_from_smiles(s)
                temp = to_dgl(g)
                temp.edata['feat'] = g.edge_attr.long()
                temp.ndata['feat'] = g.z.long()
                temp = temp.to(device)
                v = value_model(temp)
                represent.append(v)
            except:
                fp = smiles_to_fp(s)
                fp_tensor = torch.from_numpy(fp)
                fp_tensor = fp_tensor.unsqueeze(0)
                fp_tensor = fp_tensor.float().to(device)
                represent.append(fp_tensor)
        t = represent[0]
        for ind in range(1, len(represent)):
            t = torch.concat((t, represent[ind]), 0)
        t = fusion_model(t, text_embedding_tensor).to(device)
        return t.sum().item()
    else:
        try:
            g = featurize_mol_from_smiles(smi)
            temp = to_dgl(g)
            temp.edata['feat'] = g.edge_attr.long()
            temp.ndata['feat'] = g.z.long()
            temp = temp.to(device)
            v = value_model(temp)
            value = fusion_model(v, text_embedding_tensor)
            return value.sum().item()
        except:
            fp = smiles_to_fp(smi)
            fp_tensor = torch.from_numpy(fp)
            fp_tensor = fp_tensor.unsqueeze(0)
            fp_tensor = fp_tensor.float().to(device)
            value = fusion_model(fp_tensor, text_embedding_tensor)
            return value.sum().item()

def get_beam(products, beam_size):
    ins = "Please predict the reactant of the product:\n"
    inputs = reactant_tokenizer(
        ins + products[-1],
        return_tensors='pt',
        truncation=True
    ).to(device)
    with torch.no_grad():
        outputs = reactant_model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            num_beams=beam_size,
            max_new_tokens=256,
            num_return_sequences=beam_size,
            output_scores=True,
            return_dict_in_generate=True
        )

    return decode_beam_outputs(outputs["sequences"], outputs["sequences_scores"], beam_size)


def decode_beam_outputs(sequences, sequence_scores, beam_size):
    final_beams = []
    for tok, score, i in zip(sequences, sequence_scores, range(len(sequences))):
        generated_text = reactant_tokenizer.decode(tok, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        final_beams.append([generated_text, -score])

    final_beams = list(sorted(final_beams, key=lambda x: x[1]))
    answer = []
    aim_size = beam_size
    for k in range(len(final_beams)):
        if aim_size == 0:
            break
        reactants = set(final_beams[k][0].split("."))
        num_valid_reactant = 0
        sms = set()
        for r in reactants:
            m = Chem.MolFromSmiles(r)
            if m is not None:
                num_valid_reactant += 1
                sms.add(Chem.MolToSmiles(m))
        if num_valid_reactant != len(reactants):
            continue
        if len(sms):
            answer.append([sorted(list(sms)), final_beams[k][1]])
            aim_size -= 1

    return answer


def get_beam_batch(product_smiles_list, beam_size):
    if not product_smiles_list:
        return {}

    prompts = [f"Please predict the reactant of the product:\n{product_smiles}" for product_smiles in product_smiles_list]
    inputs = reactant_tokenizer(
        prompts,
        return_tensors='pt',
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = reactant_model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            num_beams=beam_size,
            max_new_tokens=256,
            num_return_sequences=beam_size,
            output_scores=True,
            return_dict_in_generate=True
        )

    batched_answers = {}
    for index, product_smiles in enumerate(product_smiles_list):
        start = index * beam_size
        end = start + beam_size
        batched_answers[product_smiles] = decode_beam_outputs(
            outputs["sequences"][start:end],
            outputs["sequences_scores"][start:end],
            beam_size
        )

    return batched_answers

def get_arguments():
    args = parse_arguments()
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    else:
        config_dict = {}

    if args.checkpoint:  # overwrite args with args from checkpoint except for the args that were contained in the config file
        arg_dict = args.__dict__
        with open(os.path.join(os.path.dirname(args.checkpoint), "train_arguments.yaml"), 'r') as arg_file:
            checkpoint_dict = yaml.load(arg_file, Loader=yaml.FullLoader)
        for key, value in checkpoint_dict.items():
            if key not in config_dict.keys():
                if isinstance(value, list):
                    for v in value:
                        arg_dict[key].append(v)
                else:
                    arg_dict[key] = value

    return args

def cano_smiles(smiles):
    try:
        tmp = Chem.MolFromSmiles(smiles)
        if tmp is None:
            return None, smiles
        tmp = Chem.RemoveHs(tmp)
        if tmp is None:
            return None, smiles
        [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
        return tmp, Chem.MolToSmiles(tmp)
    except:
        return None, smiles


def load_dataset(split):
    file_name = "data/%s_dataset.json" % split
    text_file_name = "data/text_test_dataset.json"
    dataset = []  # (product_smiles, materials_smiles, depth)
    with open(file_name, 'r') as f:
        _dataset = json.load(f)
    with open(text_file_name, 'r') as f:
        dataset_test = json.load(f)
        for _, reaction_trees in _dataset.items():
            product = reaction_trees['1']['retro_routes'][0][0].split('>')[0]
            product_mol = Chem.MolFromInchi(Chem.MolToInchi(Chem.MolFromSmiles(product)))
            product = Chem.MolToSmiles(product_mol)
            _, product = cano_smiles(product)
            for item in dataset_test:
                if item['product'] == product:
                    text = item['text']
                    intermediates = item['intermediates']
            materials_list = []
            for i in range(1, int(reaction_trees['num_reaction_trees']) + 1):
                materials_list.append(reaction_trees[str(i)]['materials'])
            dataset.append({
                "product": product,
                "targets": materials_list,
                "depth": reaction_trees['depth'],
                "text": text,
            })

    return dataset


def check_reactant_is_material(reactant):
    inchikey_prefix = smiles_to_inchikey_prefix(reactant)
    return inchikey_prefix is not None and inchikey_prefix in stock_inchikeys


def check_reactants_are_material(reactants):
    for reactant in reactants:
        inchikey_prefix = smiles_to_inchikey_prefix(reactant)
        if inchikey_prefix is None or inchikey_prefix not in stock_inchikeys:
            return False
    return True

def get_route_result(task):
    max_depth = task["depth"]
    text = task['text']
    text_embedding_tensor = text_embedding(text)
    beam_cache = {}
    value_cache = {}
    material_cache = {}
    reactants_material_cache = {}

    def get_cached_beam(route):
        expansion_mol = route[-1]
        if expansion_mol not in beam_cache:
            beam_cache[expansion_mol] = get_beam(route, args.beam_size)
        return beam_cache[expansion_mol]

    def get_cached_value(smiles):
        if smiles not in value_cache:
            value_cache[smiles] = value_fn(smiles, text_embedding_tensor)
        return value_cache[smiles]

    def is_material(smiles):
        if smiles not in material_cache:
            material_cache[smiles] = check_reactant_is_material(smiles)
        return material_cache[smiles]

    def are_materials(reactants):
        reactants_key = tuple(reactants)
        if reactants_key not in reactants_material_cache:
            reactants_material_cache[reactants_key] = check_reactants_are_material(reactants)
        return reactants_material_cache[reactants_key]

    # Initialization
    answer_set = []
    queue = []
    queue.append({
        "score": 0.0,
        "routes_info": [{"route": [task["product"]], "depth": 0}],  # List of routes information
        "starting_materials": [],
    })

    def populate_beam_cache_for_queue(queue_items):
        uncached_expansion_mols = []
        seen_expansion_mols = set()
        for queue_item in queue_items:
            routes_info = queue_item["routes_info"]
            if not routes_info:
                continue
            first_route_info = routes_info[0]
            depth = first_route_info["depth"]
            if depth > max_depth:
                continue
            expansion_mol = first_route_info["route"][-1]
            if expansion_mol in beam_cache or expansion_mol in seen_expansion_mols:
                continue
            uncached_expansion_mols.append(expansion_mol)
            seen_expansion_mols.add(expansion_mol)

        if not uncached_expansion_mols:
            return

        batch_size = max(1, args.beam_batch_size)
        for start in range(0, len(uncached_expansion_mols), batch_size):
            batch_products = uncached_expansion_mols[start:start + batch_size]
            beam_cache.update(get_beam_batch(batch_products, args.beam_size))

    while True:
        if len(queue) == 0:
            break
        populate_beam_cache_for_queue(queue)
        nxt_queue = []
        for item in queue:
            score = item["score"]
            routes_info = item["routes_info"]
            starting_materials = item["starting_materials"]
            first_route_info = routes_info[0]
            first_route, depth = first_route_info["route"], first_route_info["depth"]
            if depth > max_depth:
                continue
            expansion_mol = first_route[-1]
            expansion_cost = get_cached_value(expansion_mol)
            for expansion_solution in get_cached_beam(first_route):
                iter_routes = deepcopy(routes_info)
                iter_routes.pop(0)
                iter_starting_materials = deepcopy(starting_materials)
                expansion_reactants, reaction_cost = expansion_solution[0], expansion_solution[1]
                expansion_reactants = sorted(expansion_reactants)
                all_materials = are_materials(expansion_reactants)
                if all_materials and len(iter_routes) == 0:
                    answer_set.append({
                        "score": score + reaction_cost - expansion_cost,
                        "starting_materials": iter_starting_materials + expansion_reactants,
                    })
                else:
                    estimation_cost = 0
                    for reactant in expansion_reactants:
                        if is_material(reactant):
                            iter_starting_materials.append(reactant)
                        else:
                            estimation_cost += get_cached_value(reactant)
                            iter_routes = [{"route": first_route + [reactant], "depth": depth + 1}] + iter_routes

                    nxt_queue.append({
                        "score": score + reaction_cost + estimation_cost - expansion_cost,
                        "routes_info": iter_routes,
                        "starting_materials": iter_starting_materials
                    })
        queue = sorted(nxt_queue, key=lambda x: x["score"])[:args.beam_size]
    answer_set = sorted(answer_set, key=lambda x: x["score"])
    record_answers = set()
    final_answer_set = []
    for item in answer_set:
        score = item["score"]
        starting_materials = item["starting_materials"]
        answer_keys = []
        invalid_answer = False
        for material in starting_materials:
            inchikey_prefix = smiles_to_inchikey_prefix(material)
            if inchikey_prefix is None:
                invalid_answer = True
                break
            answer_keys.append(inchikey_prefix)
        if invalid_answer:
            continue
        if '.'.join(sorted(answer_keys)) not in record_answers:
            record_answers.add('.'.join(sorted(answer_keys)))
            final_answer_set.append({
                "score": score,
                "answer_keys": answer_keys
            })
    final_answer_set = sorted(final_answer_set, key=lambda x: x["score"])[:args.beam_size]

    # Calculate answers
    ground_truth_keys_list = []
    for targets in task["targets"]:
        target_keys = set()
        invalid_target = False
        for target in targets:
            inchikey_prefix = smiles_to_inchikey_prefix(target)
            if inchikey_prefix is None:
                invalid_target = True
                break
            target_keys.add(inchikey_prefix)
        if not invalid_target:
            ground_truth_keys_list.append(target_keys)
    for rank, answer in enumerate(final_answer_set):
        answer_keys = set(answer["answer_keys"])
        for ground_truth_keys in ground_truth_keys_list:
            if ground_truth_keys == answer_keys:
                return max_depth, rank

    return max_depth, None

def text_embedding(text):
    input_text = text
    token_text = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        return_tensors='pt'
    ).to(device)
    with torch.no_grad():
        embedding = text_model(token_text['input_ids'],attention_mask = token_text['attention_mask'])['pooler_output']
    return embedding


def summarize_depth_distribution(tasks):
    depth_counts = Counter(task["depth"] for task in tasks)
    summary = ", ".join(f"depth {depth}: {count}" for depth, count in sorted(depth_counts.items()))
    return summary if summary else "empty"


def stratified_sample_tasks(tasks, sample_size, sample_seed):
    if sample_size >= len(tasks):
        return list(tasks)

    grouped_tasks = defaultdict(list)
    for task in tasks:
        grouped_tasks[task["depth"]].append(task)

    rng = random.Random(sample_seed)
    total_tasks = len(tasks)
    allocations = {}
    remainders = []

    for depth, depth_tasks in grouped_tasks.items():
        exact_quota = sample_size * len(depth_tasks) / total_tasks
        base_quota = min(len(depth_tasks), int(exact_quota))
        allocations[depth] = base_quota
        remainders.append((exact_quota - base_quota, depth))

    if sample_size >= len(grouped_tasks):
        for depth, depth_tasks in grouped_tasks.items():
            if allocations[depth] == 0 and len(depth_tasks) > 0:
                allocations[depth] = 1

    allocated = sum(allocations.values())
    if allocated > sample_size:
        for _, depth in sorted(remainders):
            if allocated == sample_size:
                break
            if allocations[depth] > 1:
                allocations[depth] -= 1
                allocated -= 1

    remaining = sample_size - allocated
    for _, depth in sorted(remainders, reverse=True):
        if remaining == 0:
            break
        capacity = len(grouped_tasks[depth]) - allocations[depth]
        if capacity <= 0:
            continue
        take = min(capacity, remaining)
        allocations[depth] += take
        remaining -= take

    sampled_tasks = []
    for depth in sorted(grouped_tasks):
        depth_tasks = list(grouped_tasks[depth])
        rng.shuffle(depth_tasks)
        sampled_tasks.extend(depth_tasks[:allocations[depth]])

    rng.shuffle(sampled_tasks)
    return sampled_tasks

def parse_arguments():
    p = argparse.ArgumentParser()

    p.add_argument('--config', type=argparse.FileType(mode='r'), default="model/Molecule_representation/runs/PNA_qmugs_NTXentMultiplePositives_620000_123_25-08_09-19-52/12.yml")
    p.add_argument('--experiment_name', type=str, help='name that will be added to the runs folder output')
    p.add_argument('--logdir', type=str, default='runs', help='tensorboard logdirectory')
    p.add_argument('--num_epochs', type=int, default=2500, help='number of times to iterate through all samples')
    p.add_argument('--batch_size', type=int, default=1, help='samples that will be processed in parallel')
    p.add_argument('--patience', type=int, default=20, help='stop training after no improvement in this many epochs')
    p.add_argument('--minimum_epochs', type=int, default=0, help='minimum numer of epochs to run')
    p.add_argument('--dataset', type=str, default='bace_geomol', help='[qm9, zinc, drugs, geom_qm9, molhiv]')
    p.add_argument('--num_train', type=int, default=-1, help='n samples of the model samples to use for train')
    p.add_argument('--seed', type=int, default=123, help='seed for reproducibility')
    p.add_argument('--num_val', type=int, default=None, help='n samples of the model samples to use for validation')
    p.add_argument('--multithreaded_seeds', type=list, default=[],
                   help='if this is non empty, multiple threads will be started, training the same model but with the different seeds')
    p.add_argument('--seed_data', type=int, default=123, help='if you want to use a different seed for the datasplit')
    p.add_argument('--loss_func', type=str, default='MSELoss', help='Class name of torch.nn like [MSELoss, L1Loss]')
    p.add_argument('--loss_params', type=dict, default={}, help='parameters with keywords of the chosen loss function')
    p.add_argument('--critic_loss', type=str, default='MSELoss', help='Class name of torch.nn like [MSELoss, L1Loss]')
    p.add_argument('--critic_loss_params', type=dict, default={},
                   help='parameters with keywords of the chosen loss function')
    p.add_argument('--optimizer', type=str, default='Adam', help='Class name of torch.optim like [Adam, SGD, AdamW]')
    p.add_argument('--optimizer_params', type=dict, help='parameters with keywords of the chosen optimizer like lr')
    p.add_argument('--lr_scheduler', type=str,
                   help='Class name of torch.optim.lr_scheduler like [CosineAnnealingLR, ExponentialLR, LambdaLR]')
    p.add_argument('--lr_scheduler_params', type=dict, help='parameters with keywords of the chosen lr_scheduler')
    p.add_argument('--scheduler_step_per_batch', default=True, type=bool,
                   help='step every batch if true step every epoch otherwise')
    p.add_argument('--log_iterations', type=int, default=-1,
                   help='log every log_iterations iterations (-1 for only logging after each epoch)')
    p.add_argument('--expensive_log_iterations', type=int, default=100,
                   help='frequency with which to do expensive logging operations')
    p.add_argument('--eval_per_epochs', type=int, default=0,
                   help='frequency with which to do run the function run_eval_per_epoch that can do some expensive calculations on the val set or sth like that. If this is zero, then the function will never be called')
    p.add_argument('--linear_probing_samples', type=int, default=500,
                   help='number of samples to use for linear probing in the run_eval_per_epoch function of the self supervised trainer')
    p.add_argument('--num_conformers', type=int, default=3,
                   help='number of conformers to use if we are using multiple conformers on the 3d side')
    p.add_argument('--metrics', default=[], help='tensorboard metrics [mae, mae_denormalized, qm9_properties ...]')
    p.add_argument('--main_metric', default='mae_denormalized', help='for early stopping etc.')
    p.add_argument('--main_metric_goal', type=str, default='min', help='controls early stopping. [max, min]')
    p.add_argument('--val_per_batch', type=bool, default=True,
                   help='run evaluation every batch and then average over the eval results. When running the molhiv benchmark for example, this needs to be Fale because we need to evaluate on all val data at once since the metric is rocauc')
    p.add_argument('--tensorboard_functions', default=[], help='choices of the TENSORBOARD_FUNCTIONS in utils')
    p.add_argument('--checkpoint', type=str, default = "model/Molecule_representation/runs/PNA_qmugs_NTXentMultiplePositives_620000_123_25-08_09-19-52/best_checkpoint_35epochs.pt",help='path to directory that contains a checkpoint to continue training')
    p.add_argument('--pretrain_checkpoint', type=str, help='Specify path to finetune from a pretrained checkpoint')
    p.add_argument('--transfer_layers', default=[],
                   help='strings contained in the keys of the weights that are transferred')
    p.add_argument('--frozen_layers', default=[],
                   help='strings contained in the keys of the weights that are transferred')
    p.add_argument('--exclude_from_transfer', default=[],
                   help='parameters that usually should not be transferred like batchnorm params')
    p.add_argument('--transferred_lr', type=float, default=None, help='set to use a different LR for transfer layers')
    p.add_argument('--num_epochs_local_only', type=int, default=1,
                   help='when training with OptimalTransportTrainer, this specifies for how many epochs only the local predictions will get a loss')

    p.add_argument('--required_data', default=[],
                   help='what will be included in a batch like [dgl_graph, targets, dgl_graph3d]')
    p.add_argument('--collate_function', default='graph_collate', help='the collate function to use for DataLoader')
    p.add_argument('--collate_params', type=dict, default={},
                   help='parameters with keywords of the chosen collate function')
    p.add_argument('--use_e_features', default=True, type=bool, help='ignore edge features if set to False')
    p.add_argument('--targets', default=[], help='properties that should be predicted')
    p.add_argument('--device', type=str, default='cpu', help='What device to train on: cuda or cpu')

    p.add_argument('--dist_embedding', type=bool, default=False, help='add dist embedding to complete graphs edges')
    p.add_argument('--num_radial', type=int, default=6, help='number of frequencies for distance embedding')
    p.add_argument('--models_to_save', type=list, default=[],
                   help='specify after which epochs to remember the best model')

    p.add_argument('--model_type', type=str, default='PNA', help='Classname of one of the models in the models dir')
    p.add_argument('--model_parameters', type=dict, help='dictionary of model parameters')

    p.add_argument('--model3d_type', type=str, default=None, help='Classname of one of the models in the models dir')
    p.add_argument('--model3d_parameters', type=dict, help='dictionary of model parameters')
    p.add_argument('--critic_type', type=str, default=None, help='Classname of one of the models in the models dir')
    p.add_argument('--critic_parameters', type=dict, help='dictionary of model parameters')
    p.add_argument('--trainer', type=str, default='contrastive', help='[contrastive, byol, alternating, philosophy]')
    p.add_argument('--train_sampler', type=str, default=None, help='any of pytorchs samplers or a custom sampler')

    p.add_argument('--eval_on_test', type=bool, default=True, help='runs evaluation on test set if true')
    p.add_argument('--force_random_split', type=bool, default=False, help='use random split for ogb')
    p.add_argument('--reuse_pre_train_data', type=bool, default=False, help='use all data instead of ignoring that used during pre-training')
    p.add_argument('--transfer_3d', type=bool, default=False, help='set true to load the 3d network instead of the 2d network')
    args, _ = p.parse_known_args()
    return args



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mlp_dim', type=int, default=600)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--beam_size', type=int, default=5,
                        help='Beams size. Default 5. Must be 1 meaning greedy search or greater or equal 5.')
    parser.add_argument('--beam_batch_size', type=int, default=1,
                        help='Number of frontier molecules to batch together for MolT5 generation. Default 1 keeps the original behavior.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Only evaluate the first N test samples.')
    parser.add_argument('--random_limit', type=int, default=None,
                        help='Randomly evaluate N test samples.')
    parser.add_argument('--stratified_limit', type=int, default=None,
                        help='Stratified sample N test samples while preserving depth coverage.')
    parser.add_argument('--sample_seed', type=int, default=None,
                        help='Random seed used for sampled test subsets. Defaults to --seed.')
    parser.add_argument('--pretrain_checkpoint', type=str, default=FUSION_CHECKPOINT_PATH)
    parser.add_argument('--dropout', type=float, default=0.1)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cpu")

    fusion_model = FusionModel(600, 768, 600, args.n_layers, args.latent_dim, args.dropout)
    fusion_model.load_state_dict(torch.load(args.pretrain_checkpoint, map_location=device))
    fusion_model.to(device)
    fusion_model.eval()

    args_value = get_arguments()
    checkpoint = torch.load("model/Molecule_representation/runs/PNA_qmugs_NTXentMultiplePositives_620000_123_25-08_09-19-52/best_checkpoint_35epochs.pt",
        map_location=device)
    value_model = globals()[args_value.model_type](node_dim=74, edge_dim=4,
                                             **args_value.model_parameters)
    value_model.load_state_dict(checkpoint['model_state_dict'])
    value_model.to(device)
    value_model.eval()

    reactant_tokenizer = AutoTokenizer.from_pretrained(os.path.join(BASE_DIR, 'model', 'MolT5'), use_fast=False)
    reactant_model = T5ForConditionalGeneration.from_pretrained(os.path.join(BASE_DIR, 'model', 'MolT5'))
    reactant_model.to(device)
    reactant_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(BASE_DIR, 'model', 'scibert'))
    text_model = AutoModel.from_pretrained(os.path.join(BASE_DIR, 'model', 'scibert'))
    text_model.to(device)
    text_model.eval()

    stock = pd.read_hdf(os.path.join(BASE_DIR, "data", "zinc_stock_17_04_20.hdf5"), key="table")
    stockinchikey_list = stock.inchi_key.values
    stock_inchikeys = set([x[:14] for x in stockinchikey_list])

    tasks = load_dataset('test')
    sampling_args = [args.limit is not None, args.random_limit is not None, args.stratified_limit is not None]
    if sum(sampling_args) > 1:
        raise ValueError("Only one of --limit, --random_limit, or --stratified_limit can be specified.")

    if args.stratified_limit is not None:
        sample_size = min(args.stratified_limit, len(tasks))
        sample_seed = args.seed if args.sample_seed is None else args.sample_seed
        tasks = stratified_sample_tasks(tasks, sample_size, sample_seed)
        print(f"Stratified sampled {sample_size} test samples with seed {sample_seed}.", flush=True)
        print(f"Sample depth distribution: {summarize_depth_distribution(tasks)}", flush=True)
    elif args.random_limit is not None:
        sample_size = min(args.random_limit, len(tasks))
        sample_seed = args.seed if args.sample_seed is None else args.sample_seed
        sampler = random.Random(sample_seed)
        tasks = sampler.sample(tasks, sample_size)
        print(f"Randomly sampled {sample_size} test samples with seed {sample_seed}.", flush=True)
    elif args.limit is not None:
        tasks = tasks[:args.limit]
    overall_result = np.zeros((args.beam_size, 2))
    depth_hit = np.zeros((2, 15, args.beam_size))

    for epoch in trange(0, len(tasks)):
        max_depth, rank = get_route_result(tasks[epoch])
        overall_result[:, 1] += 1
        depth_hit[1, max_depth, :] += 1
        if rank is not None:
            overall_result[rank:, 0] += 1
            depth_hit[0, max_depth, rank:] += 1
    print("overall_result: ", overall_result, 100 * overall_result[:, 0] / overall_result[:, 1], flush=True)
    print("depth_hit: ", depth_hit, 100 * depth_hit[0, :, :] / depth_hit[1, :, :])

