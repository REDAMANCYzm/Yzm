import torch
from torch.utils.data import Dataset, DataLoader, Subset
import json
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import ChiralType
from torch_scatter import scatter
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dgl
from transformers import AutoModel,AutoTokenizer
import argparse
import yaml
import os
from datetime import datetime
import torch.nn as nn
import logging
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from model.Molecule_representation.commons.losses import *
from model.Molecule_representation.models import *
from model.Molecule_representation.datasets.samplers import *
from training_curve_png import default_curve_path as default_curve_png_path, save_training_curves_png

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FUSION_CHECKPOINT_PATH = os.path.join(
    BASE_DIR,
    'model',
    'value_function_text_conditioned_fusion-model_val.pkl'
)
DEFAULT_CACHE_PATH = os.path.join(
    BASE_DIR,
    'data',
    'fusion_train_embeddings_scibert_pna.pt'
)

bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
types = {'H': 0, 'Li': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 5, 'F': 6, 'Na': 7, 'Mg': 8, 'Al': 9, 'Si': 10,
               'P': 11, 'S': 12, 'Cl': 13, 'K': 14, 'Ca': 15, 'V': 16, 'Cr': 17, 'Mn': 18, 'Cu': 19, 'Zn': 20,
               'Ga': 21, 'Ge': 22, 'As': 23, 'Se': 24, 'Br': 25, 'Ag': 26, 'In': 27, 'Sb': 28, 'I': 29, 'Gd': 30,
               'Pt': 31, 'Au': 32, 'Hg': 33, 'Bi': 34}
dihedral_pattern = Chem.MolFromSmarts('[*]~[*]~[*]~[*]')
chirality = {ChiralType.CHI_TETRAHEDRAL_CW: -1.,
             ChiralType.CHI_TETRAHEDRAL_CCW: 1.,
             ChiralType.CHI_UNSPECIFIED: 0,
             ChiralType.CHI_OTHER: 0}

def one_k_encoding(value, choices):
    """
    Creates a one-hot encoding with an extra category for uncommon values.
    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def open_log_file(log_path):
    if not log_path:
        return None
    absolute_path = os.path.abspath(log_path)
    log_dir = os.path.dirname(absolute_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    log_file = open(absolute_path, 'a', encoding='utf-8')
    log_file.write(f"\n===== Run started at {datetime.now().isoformat(timespec='seconds')} =====\n")
    log_file.flush()
    return log_file


def log_message(message, log_file=None):
    print(message)
    if log_file is not None:
        log_file.write(f"{message}\n")
        log_file.flush()


def default_curve_path(log_path):
    return default_curve_png_path(log_path)


def _resolve_curve_path(log_path, curve_path):
    if curve_path:
        return os.path.abspath(curve_path)
    if log_path:
        return default_curve_path(log_path)
    return None


class RawFusionDataset(Dataset):
    def __init__(self, file_name, tokenizer, text_model, value_model, device):
        with open(file_name, 'r') as f:
            self.dataset = json.load(f)
        self.tokenizer = tokenizer
        self.text_model = text_model
        self.value_model = value_model
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def featurize_mol_from_smiles(self,smiles):
        # filter fragments
        if '.' in smiles:
            raise Exception
            return None

        # filter mols rdkit can't intrinsically handle
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()

        type_idx = []
        atomic_number = []
        atom_features = []
        chiral_tag = []
        neighbor_dict = {}
        ring = mol.GetRingInfo()
        for i, atom in enumerate(mol.GetAtoms()):
            type_idx.append(types[atom.GetSymbol()])
            n_ids = [n.GetIdx() for n in atom.GetNeighbors()]
            if len(n_ids) > 1:
                neighbor_dict[i] = torch.tensor(n_ids)
            chiral_tag.append(chirality[atom.GetChiralTag()])
            atomic_number.append(atom.GetAtomicNum())
            atom_features.extend([atom.GetAtomicNum(),
                                  1 if atom.GetIsAromatic() else 0])
            atom_features.extend(one_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]))
            atom_features.extend(one_k_encoding(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2]))
            atom_features.extend(one_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]))
            atom_features.extend(one_k_encoding(atom.GetFormalCharge(), [-1, 0, 1]))
            atom_features.extend([int(ring.IsAtomInRingOfSize(i, 3)),
                                  int(ring.IsAtomInRingOfSize(i, 4)),
                                  int(ring.IsAtomInRingOfSize(i, 5)),
                                  int(ring.IsAtomInRingOfSize(i, 6)),
                                  int(ring.IsAtomInRingOfSize(i, 7)),
                                  int(ring.IsAtomInRingOfSize(i, 8))])
            atom_features.extend(one_k_encoding(int(ring.NumAtomRings(i)), [0, 1, 2, 3]))

        z = torch.tensor(atomic_number, dtype=torch.long)
        chiral_tag = torch.tensor(chiral_tag, dtype=torch.float)

        row, col, edge_type, bond_features = [], [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_type += 2 * [bonds[bond.GetBondType()]]
            bt = tuple(
                sorted(
                    [bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()])), bond.GetBondTypeAsDouble()
            bond_features += 2 * [int(bond.IsInRing()),
                                  int(bond.GetIsConjugated()),
                                  int(bond.GetIsAromatic())]

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)
        # bond_features = torch.tensor(bond_features, dtype=torch.float).view(len(bond_type), -1)

        perm = (edge_index[0] * N + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_type = edge_type[perm]
        # edge_attr = torch.cat([edge_attr[perm], bond_features], dim=-1)
        edge_attr = edge_attr[perm]

        row, col = edge_index
        hs = (z == 1).to(torch.float)
        num_hs = scatter(hs[row], col, dim_size=N).tolist()

        x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
        x2 = torch.tensor(atom_features).view(N, -1)
        x = torch.cat([x1.to(torch.float), x2], dim=-1)

        data = Data(z=x, edge_index=edge_index, edge_attr=edge_attr, neighbors=neighbor_dict, chiral_tag=chiral_tag,
                    name=smiles, num_nodes=N)
        return data

    def represent(self, smi):
        if '.' in smi:
            represent = []
            for s in smi.split('.'):
                g = self.featurize_mol_from_smiles(s)
                temp = to_dgl(g)
                temp.edata['feat'] = g.edge_attr.long()
                temp.ndata['feat'] = g.z.long()
                with torch.no_grad():
                    v = self.value_model(temp).to(self.device)
                represent.append(v)
            t = represent[0]
            for ind in range(1, len(represent)):
                t = torch.concat((t, represent[ind]), 0)
            return t
        else:
            g = self.featurize_mol_from_smiles(smi)
            temp = to_dgl(g)
            temp.edata['feat'] = g.edge_attr.long()
            temp.ndata['feat'] = g.z.long()
            with torch.no_grad():
                v = self.value_model(temp).to(self.device)
            return v

    def text_embedding(self, text):
        token_text = self.tokenizer(text, max_length=512, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            embedding = self.text_model(
                token_text['input_ids'],
                attention_mask=token_text['attention_mask']
            )['pooler_output']
        return embedding

    def __getitem__(self, idx):
        item = self.dataset[idx]
        molecule_representation = self.represent(item['product']).clone().detach().float()
        text_embedding = self.text_embedding(item['text']).clone().detach().float()
        cost = torch.tensor(item['cost'], dtype=torch.float32)
        return molecule_representation, text_embedding, cost

    def get_depths(self):
        return [item.get('depth', 0) for item in self.dataset]


class CachedEmbeddingDataset(Dataset):
    def __init__(self, cache_payload):
        self.molecule_embeddings = cache_payload['molecule_embeddings'].float()
        self.text_embeddings = cache_payload['text_embeddings'].float()
        self.costs = cache_payload['costs'].float()
        self.depths = cache_payload['depths'].long()
        self.products = cache_payload['products']
        self.texts = cache_payload['texts']

    def __len__(self):
        return self.costs.size(0)

    def __getitem__(self, idx):
        return (
            self.molecule_embeddings[idx],
            self.text_embeddings[idx],
            self.costs[idx]
        )

    def get_depths(self):
        return self.depths.tolist()


def squeeze_embedding(embedding, expected_dim, field_name, identifier):
    embedding = embedding.detach().cpu()
    if embedding.dim() == 2 and embedding.size(0) == 1:
        embedding = embedding.squeeze(0)
    if embedding.dim() != 1 or embedding.size(0) != expected_dim:
        raise ValueError(
            f"Expected {field_name} for {identifier} to have shape [{expected_dim}] or [1, {expected_dim}], "
            f"got {tuple(embedding.shape)}"
        )
    return embedding.float()


def build_embedding_cache(raw_dataset, cache_path, dataset_path, scibert_dir, pna_checkpoint_path):
    molecule_embeddings = []
    text_embeddings = []
    costs = []
    depths = []
    products = []
    texts = []

    for index in tqdm(range(len(raw_dataset)), desc="Building embedding cache"):
        item = raw_dataset.dataset[index]
        product = item['product']
        text = item['text']

        molecule_embedding = squeeze_embedding(
            raw_dataset.represent(product),
            expected_dim=600,
            field_name='molecule embedding',
            identifier=product
        )
        text_embedding = squeeze_embedding(
            raw_dataset.text_embedding(text),
            expected_dim=768,
            field_name='text embedding',
            identifier=product
        )

        molecule_embeddings.append(molecule_embedding)
        text_embeddings.append(text_embedding)
        costs.append(float(item['cost']))
        depths.append(int(item.get('depth', 0)))
        products.append(product)
        texts.append(text)

    payload = {
        'molecule_embeddings': torch.stack(molecule_embeddings, dim=0),
        'text_embeddings': torch.stack(text_embeddings, dim=0),
        'costs': torch.tensor(costs, dtype=torch.float32),
        'depths': torch.tensor(depths, dtype=torch.long),
        'products': products,
        'texts': texts,
        'meta': {
            'dataset_path': os.path.abspath(dataset_path),
            'num_samples': len(products),
            'scibert_dir': os.path.abspath(scibert_dir),
            'pna_checkpoint_path': os.path.abspath(pna_checkpoint_path),
        }
    }

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def validate_cache_payload(cache_payload, dataset_path, scibert_dir, pna_checkpoint_path):
    required_keys = {
        'molecule_embeddings',
        'text_embeddings',
        'costs',
        'depths',
        'products',
        'texts',
        'meta',
    }
    missing_keys = required_keys.difference(cache_payload.keys())
    if missing_keys:
        raise ValueError(f"Cache file is missing required keys: {sorted(missing_keys)}")

    meta = cache_payload['meta']
    expected_meta = {
        'dataset_path': os.path.abspath(dataset_path),
        'scibert_dir': os.path.abspath(scibert_dir),
        'pna_checkpoint_path': os.path.abspath(pna_checkpoint_path),
    }
    for key, expected_value in expected_meta.items():
        actual_value = meta.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"Cache meta mismatch for {key}: expected {expected_value}, got {actual_value}. "
                f"Please rebuild the cache."
            )

    num_samples = meta.get('num_samples')
    actual_num_samples = cache_payload['costs'].size(0)
    if num_samples != actual_num_samples:
        raise ValueError(
            f"Cache num_samples mismatch: meta says {num_samples}, tensors contain {actual_num_samples}. "
            f"Please rebuild the cache."
        )

    if cache_payload['molecule_embeddings'].size(0) != actual_num_samples:
        raise ValueError("Molecule embedding count does not match costs count in cache.")
    if cache_payload['text_embeddings'].size(0) != actual_num_samples:
        raise ValueError("Text embedding count does not match costs count in cache.")
    if cache_payload['depths'].size(0) != actual_num_samples:
        raise ValueError("Depth count does not match costs count in cache.")
    if len(cache_payload['products']) != actual_num_samples:
        raise ValueError("Product count does not match costs count in cache.")
    if len(cache_payload['texts']) != actual_num_samples:
        raise ValueError("Text count does not match costs count in cache.")


def build_stratified_train_val_subsets(dataset, val_ratio, seed):
    if val_ratio <= 0:
        indices = list(range(len(dataset)))
        return Subset(dataset, indices), None, indices, []

    grouped_indices = defaultdict(list)
    for index, depth in enumerate(dataset.get_depths()):
        grouped_indices[depth].append(index)

    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []

    for depth in sorted(grouped_indices):
        depth_indices = list(grouped_indices[depth])
        rng.shuffle(depth_indices)

        if len(depth_indices) <= 1:
            train_indices.extend(depth_indices)
            continue

        proposed_val_size = int(round(len(depth_indices) * val_ratio))
        val_size = max(1, proposed_val_size)
        val_size = min(len(depth_indices) - 1, val_size)

        val_indices.extend(depth_indices[:val_size])
        train_indices.extend(depth_indices[val_size:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return Subset(dataset, train_indices), Subset(dataset, val_indices), train_indices, val_indices


def evaluate_model(fusion_model, dataloader, criterion, device):
    fusion_model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for molecule_embeddings, text_embeddings, costs in dataloader:
            molecule_embeddings = molecule_embeddings.to(device)
            text_embeddings = text_embeddings.to(device)
            costs = costs.to(device)

            outputs = fusion_model(molecule_embeddings, text_embeddings)
            costs = costs.view_as(outputs)
            batch_size = costs.size(0)

            total_loss += criterion(outputs, costs).item() * batch_size
            total_mae += torch.abs(outputs - costs).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        return None, None

    return total_loss / total_samples, total_mae / total_samples

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
        # layers.append(nn.BatchNorm1d(latent_dim,
        #                              track_running_stats=False))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))
        for _ in range(self.n_layers - 1):
            layers.append(nn.Linear(latent_dim, latent_dim))
            # layers.append(nn.BatchNorm1d(latent_dim,
            #                              track_running_stats=False))
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
        return value_output

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

def parse_arguments():
    p = argparse.ArgumentParser(allow_abbrev=False)

    p.add_argument('--config', type=argparse.FileType(mode='r'), default=os.path.join(BASE_DIR, "model", "Molecule_representation", "runs", "PNA_qmugs_NTXentMultiplePositives_620000_123_25-08_09-19-52", "12.yml"))
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
    p.add_argument('--checkpoint', type=str, default = os.path.join(BASE_DIR, "model", "Molecule_representation", "runs", "PNA_qmugs_NTXentMultiplePositives_620000_123_25-08_09-19-52", "best_checkpoint_35epochs.pt"),help='path to directory that contains a checkpoint to continue training')
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    # ===================== model ====================== #
    parser.add_argument('--fp_dim', type=int, default=600)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--latent_dim', type=int, default=128)

    # ==================== training ==================== #
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--val_seed', type=int, default=42)
    parser.add_argument('--build_cache_only', action='store_true')
    parser.add_argument('--cache_path', type=str, default=DEFAULT_CACHE_PATH)
    parser.add_argument('--overwrite_cache', action='store_true')
    parser.add_argument('--log_file', type=str, default=None)
    parser.add_argument('--curve_path', type=str, default=None,
                        help='Optional PNG path for training curves. Defaults next to --log_file.')
    # parser.add_argument('--pretrain_checkpoint', type=str, default='value_function_fusion-model.pkl')


    args = parser.parse_args()
    if not 0 <= args.val_ratio < 1:
        raise ValueError(f"--val_ratio must be in [0, 1), got {args.val_ratio}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache_path = os.path.abspath(args.cache_path)
    log_file = open_log_file(args.log_file)
    dataset_path = os.path.join(BASE_DIR, "data", "fusion-model_traindataset.json")
    scibert_dir = os.path.join(BASE_DIR, 'model', 'scibert')
    pna_checkpoint_path = os.path.join(
        BASE_DIR,
        "model",
        "Molecule_representation",
        "runs",
        "PNA_qmugs_NTXentMultiplePositives_620000_123_25-08_09-19-52",
        "best_checkpoint_35epochs.pt"
    )

    tokenizer = AutoTokenizer.from_pretrained(scibert_dir)
    text_model = AutoModel.from_pretrained(scibert_dir)
    text_model.to(device)
    text_model.eval()

    args_value = get_arguments()
    checkpoint = torch.load(pna_checkpoint_path, map_location=device)
    value_model = globals()[args_value.model_type](node_dim=74, edge_dim=4,
                                                   **args_value.model_parameters)
    value_model.load_state_dict(checkpoint['model_state_dict'])
    value_model.eval()

    if args.build_cache_only and os.path.exists(cache_path) and not args.overwrite_cache:
        raise FileExistsError(
            f"Cache already exists at {cache_path}. Use --overwrite_cache to rebuild it."
        )

    if args.build_cache_only:
        raw_dataset = RawFusionDataset(dataset_path, tokenizer, text_model, value_model, device)
        cache_payload = build_embedding_cache(
            raw_dataset,
            cache_path=cache_path,
            dataset_path=dataset_path,
            scibert_dir=scibert_dir,
            pna_checkpoint_path=pna_checkpoint_path
        )
        log_message(f"Embedding cache saved to: {cache_path}", log_file)
        log_message(f"Cached samples: {cache_payload['costs'].size(0)}", log_file)
        if log_file is not None:
            log_file.close()
        raise SystemExit(0)

    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Embedding cache not found at {cache_path}. "
            f"Run `python Fusion_model.py --build_cache_only` first."
        )

    cache_payload = torch.load(cache_path, map_location='cpu')
    validate_cache_payload(
        cache_payload,
        dataset_path=dataset_path,
        scibert_dir=scibert_dir,
        pna_checkpoint_path=pna_checkpoint_path
    )

    dataset = CachedEmbeddingDataset(cache_payload)
    train_dataset, val_dataset, train_indices, val_indices = build_stratified_train_val_subsets(
        dataset,
        args.val_ratio,
        args.val_seed
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = None
    if val_dataset is not None and len(val_indices) > 0:
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    log_message(f"Total samples: {len(dataset)}", log_file)
    log_message(f"Train samples: {len(train_indices)}", log_file)
    log_message(f"Validation samples: {len(val_indices)}", log_file)
    log_message(f"Validation ratio: {args.val_ratio}", log_file)
    log_message(f"Validation split seed: {args.val_seed}", log_file)
    log_message(f"Embedding cache: {cache_path}", log_file)
    if args.log_file:
        log_message(f"Training log: {os.path.abspath(args.log_file)}", log_file)
    log_message(
        (
            f"Hyperparameters: batch_size={args.batch_size}, n_epochs={args.n_epochs}, "
            f"lr={args.lr}, dropout={args.dropout}, latent_dim={args.latent_dim}, "
            f"n_layers={args.n_layers}, seed={args.seed}"
        ),
        log_file
    )

    fusion_model = FusionModel(600, 768, 600, args.n_layers, args.latent_dim, args.dropout)
    # fusion_model.load_state_dict(torch.load(args.pretrain_checkpoint))
    fusion_model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(fusion_model.parameters(), lr=args.lr)

    best_metric = float('inf')
    history = []
    for epoch in range(args.n_epochs):
        fusion_model.train()
        running_loss = 0.0
        running_mae = 0.0
        seen_samples = 0
        for batch in tqdm(train_dataloader):
            molecule_embeddings, text_embeddings, costs = batch
            molecule_embeddings = molecule_embeddings.to(device)
            text_embeddings = text_embeddings.to(device)
            costs = costs.to(device)

            optimizer.zero_grad()
            outputs = fusion_model(molecule_embeddings, text_embeddings)
            costs = costs.view_as(outputs)
            loss = criterion(outputs, costs)
            loss.backward()
            optimizer.step()

            batch_size = costs.size(0)
            running_loss += loss.item() * batch_size
            running_mae += torch.abs(outputs.detach() - costs).sum().item()
            seen_samples += batch_size

        epoch_loss = running_loss / seen_samples
        epoch_mae = running_mae / seen_samples
        epoch_record = {
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'train_mae': epoch_mae,
            'val_loss': None,
            'val_mae': None
        }

        if val_dataloader is not None:
            val_loss, val_mae = evaluate_model(fusion_model, val_dataloader, criterion, device)
            epoch_record['val_loss'] = val_loss
            epoch_record['val_mae'] = val_mae
            log_message(
                f"\nEpoch [{epoch + 1}/{args.n_epochs}], "
                f"Train Loss: {epoch_loss:.4f}, Train MAE: {epoch_mae:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}",
                log_file
            )
            current_metric = val_loss
        else:
            log_message(
                f"\nEpoch [{epoch + 1}/{args.n_epochs}], "
                f"Train Loss: {epoch_loss:.4f}, Train MAE: {epoch_mae:.4f}",
                log_file
            )
            current_metric = epoch_loss

        history.append(epoch_record)

        if current_metric < best_metric:
            best_metric = current_metric
            torch.save(fusion_model.state_dict(), FUSION_CHECKPOINT_PATH)
            if val_dataloader is not None:
                log_message(f"Best model saved with validation loss: {best_metric:.4f}", log_file)
            else:
                log_message(f"Best model saved with training loss: {best_metric:.4f}", log_file)

    curve_path = _resolve_curve_path(args.log_file, args.curve_path)
    if curve_path and history:
        save_training_curves_png(history, curve_path, title='Fusion Model Training Curves')
        log_message(f"Training curves saved to: {curve_path}", log_file)

    if log_file is not None:
        log_file.close()
