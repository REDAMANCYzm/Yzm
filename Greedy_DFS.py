import argparse
import os
import random
import json
import torch
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from rdkit import Chem
from rdkit.rdBase import DisableLog
from transformers import AutoModel,AutoTokenizer,T5ForConditionalGeneration

DisableLog('rdApp.warning')
DisableLog('rdApp.error')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
reactant_tokenizer = None
reactant_model = None


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


def get_beam(products, beam_size):
    ins = "Please predict the reactant of the product:\n"
    final_beams = []
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

    for tok, score,i in zip(outputs["sequences"], outputs["sequences_scores"],range(len(outputs["sequences"]))):
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


def load_dataset(split):
    file_name = os.path.join(BASE_DIR, "data", f"{split}_dataset.json")
    dataset = [] # (product_smiles, materials_smiles, depth)
    with open(file_name, 'r') as f:
        _dataset = json.load(f)
        for _, reaction_trees in _dataset.items():
            product = reaction_trees['1']['retro_routes'][0][0].split('>')[0]
            product_mol = Chem.MolFromInchi(Chem.MolToInchi(Chem.MolFromSmiles(product)))
            product = Chem.MolToSmiles(product_mol)
            _, product = cano_smiles(product)
            materials_list = []
            for i in range(1, int(reaction_trees['num_reaction_trees'])+1):
                materials_list.append(reaction_trees[str(i)]['materials'])
            dataset.append({
                "product": product, 
                "targets": materials_list, 
                "depth": reaction_trees['depth']
            })

    return dataset


def smiles_to_inchikey_prefix(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToInchiKey(mol)[:14]


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
    # Initialization
    answer_list = []
    queue = {
        "routes_info": [{"route": [task["product"]], "depth": 0}],  # List of routes information
        "starting_materials": [],
    }
    while True:
        if len(queue) == 0:
            break
        nxt_queue = {}
        routes_info = queue["routes_info"]
        starting_materials = queue["starting_materials"]
        first_route_info = routes_info[0]
        first_route, depth = first_route_info["route"], first_route_info["depth"]
        if depth > max_depth:
            break
        if len(get_beam(first_route, args.beam_size)) == 0:
            break
        expansion_solution = get_beam(first_route, args.beam_size)[0]
        iter_routes = deepcopy(routes_info)
        iter_routes.pop(0)
        iter_starting_materials = deepcopy(starting_materials)
        expansion_reactants, _ = expansion_solution[0], expansion_solution[1]
        expansion_reactants = sorted(expansion_reactants)
        if check_reactants_are_material(expansion_reactants) and len(iter_routes) == 0:
            answer_list = iter_starting_materials+expansion_reactants
        else:
            for reactant in expansion_reactants:
                if check_reactant_is_material(reactant):
                    iter_starting_materials.append(reactant)
                else:
                    iter_routes = [{"route": first_route+[reactant], "depth": depth+1}] + iter_routes
            nxt_queue = {
                "routes_info": iter_routes,
                "starting_materials": iter_starting_materials
            }
        queue = nxt_queue
            
    answer_keys = set()
    for material in answer_list:
        inchikey_prefix = smiles_to_inchikey_prefix(material)
        if inchikey_prefix is None:
            return max_depth, False
        answer_keys.add(inchikey_prefix)

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

    for ground_truth_keys in ground_truth_keys_list:
        if ground_truth_keys == answer_keys:
            return max_depth, True

    return max_depth, False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--max_length', type=int, default=200, help='The max length of a molecule.')
    parser.add_argument('--max_depth', type=int, default=14, help='The max depth of a synthesis route.')
    parser.add_argument('--embedding_size', type=int, default=64, help='The size of embeddings')
    parser.add_argument('--hidden_size', type=int, default=640, help='The size of hidden units')
    parser.add_argument('--num_hidden_layers', type=int, default=3, help='Number of layers in encoder\'s module. Default 3.')
    parser.add_argument('--num_attention_heads', type=int, default=10, help='Number of attention heads. Default 10.')
    parser.add_argument('--intermediate_size', type=int, default=512, help='The size of hidden units of position-wise layer.')
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--temperature', type=float, default=1.2, help='Temperature for decoding. Default 1.2')
    parser.add_argument('--beam_size', type=int, default=5, help='Beams size. Default 5. Must be 1 meaning greedy search or greater or equal 5.')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    reactant_tokenizer = AutoTokenizer.from_pretrained(os.path.join(BASE_DIR, "model", "MolT5"), use_fast=False)
    reactant_model = T5ForConditionalGeneration.from_pretrained(os.path.join(BASE_DIR, "model", "MolT5"))
    reactant_model.to(device)
    reactant_model.eval()

    stock = pd.read_hdf(os.path.join(BASE_DIR, "data", "zinc_stock_17_04_20.hdf5"), key="table")
    stockinchikey_list = stock.inchi_key.values
    stock_inchikeys = set([x[:14] for x in stockinchikey_list])

    tasks = load_dataset('test')
    overall_result = np.zeros((2))
    depth_hit = np.zeros((2, 15))
    for epoch in tqdm(range(0, len(tasks))):
        max_depth, match = get_route_result(tasks[epoch])
        overall_result[1] += 1
        depth_hit[1, max_depth] += 1
        if match:
            overall_result[0] += 1
            depth_hit[0, max_depth] += 1
        print("overall_result: ", overall_result, 100 * overall_result[0] / overall_result[1])

    # print("overall_result: ", overall_result, 100 * overall_result[0] / overall_result[1])
    print("depth_hit: ", depth_hit, 100 * depth_hit[0, :] / depth_hit[1, :])
