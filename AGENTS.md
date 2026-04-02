# AGENTS.md

This file provides guidance to Qoder (qoder.com) when working with code in this repository.

## Project Overview

RetroInText is a multimodal Large Language Model (LLM) framework for retrosynthetic planning. It combines molecular representations with text descriptions of synthetic routes to predict reactants from products.

## Environment Setup

This project uses Python 3.8.19 with Conda. Set up the environment using:

```bash
conda env create -f environment.yaml
conda activate MS
```

Key dependencies include PyTorch 2.2.0, DGL 2.1.0, transformers 4.39.3, and RDKit.

## Project Structure

```
.
├── main.py                    # Main entry point for running experiments on RetroBench dataset
├── Fusion_model.py            # Training script for the attention-based fusion model
├── Greedy_DFS.py              # Greedy DFS baseline implementation
├── verify_fusion.py           # Verification script for fusion model
├── run_translation/           # MolT5 fine-tuning for molecule-text translation
│   └── run_translation.py
├── dataprocess/               # Data preprocessing scripts
│   ├── train_text_generation.py   # Generate text descriptions using OpenAI API
│   ├── test_text_generation.py
│   ├── get_reaction_cost.py
│   └── get_cost.py
├── model/
│   ├── Molecule_representation/   # 3D molecular representation learning models
│   │   ├── models/                # GNN models (PNA, GIN, EGNN, etc.)
│   │   ├── datasets/              # Dataset loaders
│   │   ├── commons/               # Utilities and losses
│   │   └── runs/                  # Pre-trained checkpoints
│   ├── MolT5/                     # Fine-tuned MolT5 model for single-step prediction
│   ├── scibert/                   # SciBERT model for text encoding
│   └── value_function_fusion-model.pkl  # Pre-trained fusion model checkpoint
└── data/                        # Dataset directory
    ├── train_dataset.json
    ├── test_dataset.json
    ├── text_train_dataset.json
    ├── text_test_dataset.json
    ├── fusion-model_traindataset.json
    └── zinc_stock_17_04_20.hdf5   # Building block stock for route evaluation
```

## Common Commands

### Running Experiments

```bash
# Run main experiment (beam search with size 5, seed 42)
python main.py --beam_size 5 --seed 42

# Run Greedy DFS baseline
python Greedy_DFS.py --beam_size 5 --seed 42
```

### Training

```bash
# Train the fusion model (requires pre-trained models first)
python Fusion_model.py --n_epochs 20 --batch_size 64 --lr 1e-3

# Fine-tune MolT5 (requires MolT5 base model in run_translation/model/)
cd run_translation
python run_translation.py --Fine-tune.txt
```

### Data Processing

```bash
cd dataprocess

# Generate text descriptions for training data (requires OpenAI API key)
python train_text_generation.py

# Generate text descriptions for test data
python test_text_generation.py

# Calculate reaction costs
python get_reaction_cost.py

# Calculate total costs
python get_cost.py
```

## Model Architecture

The system consists of three main components:

1. **Single-step Model (MolT5)**: A fine-tuned T5 model that predicts reactants from product SMILES. Located in `model/MolT5/`. It takes a product SMILES string and generates reactant SMILES.

2. **Molecule Representation Model (PNA)**: A Principal Neighbourhood Aggregation GNN that encodes molecules into 600-dimensional vectors using 3D conformers. Located in `model/Molecule_representation/`. The pre-trained checkpoint is at `model/Molecule_representation/runs/PNA_qmugs_NTXentMultiplePositives_620000_123_25-08_09-19-52/best_checkpoint_35epochs.pt`.

3. **Fusion Model**: An attention-based module that fuses molecule embeddings (600-dim) with text embeddings (768-dim from SciBERT) to predict synthesis costs. Defined in `Fusion_model.py` and `main.py`. The attention mechanism uses text embeddings as Query/Key and molecule embeddings as Value.

## Key Implementation Details

- **Value Function**: The `value_fn()` in `main.py` computes the estimated cost of synthesizing a molecule. It uses the fusion model to combine PNA molecule representations with SciBERT text embeddings.

- **Beam Search**: Implemented in `get_beam()` in `main.py`. Uses MolT5 to generate candidate reactants, then scores routes using the value function.

- **Stock Check**: `check_reactant_is_material()` verifies if a molecule is available in the ZINC stock file (`data/zinc_stock_17_04_20.hdf5`) using InChIKey prefix matching (first 14 characters).

- **Dataset Format**: JSON files containing product SMILES, target materials, synthesis depth, and text descriptions. See `data/test_dataset.json` and `data/text_test_dataset.json` for examples.

## Required Pre-trained Models

Before running experiments, ensure these models are downloaded and placed in the correct locations:

1. **MolT5**: Place at `model/MolT5/` (fine-tuned) or download from HuggingFace for base model
2. **SciBERT**: Place at `model/scibert/`
3. **PNA Checkpoint**: Place at `model/Molecule_representation/runs/PNA_qmugs_NTXentMultiplePositives_620000_123_25-08_09-19-52/best_checkpoint_35epochs.pt`
4. **Fusion Model**: `model/value_function_fusion-model.pkl` (or train using `Fusion_model.py`)

See README.md for download links.

## Important File Paths

- Config file: `model/Molecule_representation/runs/PNA_qmugs_NTXentMultiplePositives_620000_123_25-08_09-19-52/12.yml`
- Stock file: `data/zinc_stock_17_04_20.hdf5`
- Test dataset: `data/test_dataset.json`
- Text test dataset: `data/text_test_dataset.json`
