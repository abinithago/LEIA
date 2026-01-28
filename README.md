# LEIA: Low-rank Error-Informed Adjustment

LEIA (Low-rank Error-Informed Adjustment) is a method for learning low-rank adjustments to base classifiers by identifying error-informed subspaces and learning adjustments within those subspaces.

## Overview

LEIA improves upon ERM (Empirical Risk Minimization) by:

1. **Identifying Error Subspace**: Computes error-weighted covariance matrix from embeddings and extracts top-k eigenvectors (the error subspace)
2. **Learning Low-Rank Adjustment**: Learns a small adjustment matrix in the error subspace to correct classifier predictions
3. **Preserving Base Performance**: Only adjusts predictions along error directions, preserving correct predictions

## Installation

### Setup

1. Clone or download this directory

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate leia_env
```

The `environment.yml` file includes all required dependencies:
- Python 3.9.7
- PyTorch 1.13.0 and torchvision 0.14.0
- transformers 4.24.0 (for BERT models)
- wilds 2.0.0 (for CivilComments dataset)
- numpy, pandas, scikit-learn, scipy
- Other utilities (tqdm, pyyaml, wandb, matplotlib, etc.)

3. **Set up your datasets**:
   - Point to your dataset directories (see Dataset Setup below)
   - **Metadata is automatically generated** from raw data - no manual formatting needed!

## Directory Structure

```
leia/                         # Package root
├── data/                     # Dataset loaders and transforms
│   ├── datasets.py           # Dataset classes (SpuriousDataset, MultiNLIDataset, etc.)
│   ├── transforms.py         # Data augmentation transforms
│   ├── augmix.py             # AugMix augmentation
│   └── generate_metadata.py  # Auto-generate metadata from raw datasets
├── losses/                   # Loss functions
│   └── __init__.py           # LEIA weights, spectral decomposition, loss functions
├── models/                   # Model architectures
│   ├── __init__.py           # LEIAModel class
│   └── architectures.py      # ResNet50, BERT model factories
├── optimizers/               # Optimizer and scheduler factories
├── utils/                    # Utilities (metrics, logging, training, embedding extraction)
├── scripts/                  # Training and extraction scripts
│   ├── train_stage1.py       # Stage 1: Train ERM base classifier
│   ├── extract_embeddings.py # Extract embeddings from Stage 1 model
│   └── train.py              # Stage 2: Train LEIA/ERM on embeddings
├── config.py                 # Configuration and dataset paths
├── generate_metadata.py      # CLI tool for manual metadata generation
├── environment.yml           # Conda environment definition
├── README.md                 # This file
└── SETUP.md                  # Detailed setup guide
```

## Usage

LEIA follows a two-stage training paradigm:

### Stage 1: Train ERM Base Classifier

Train a standard ERM classifier on your dataset:

```bash
python scripts/train_stage1.py \
    --dataset waterbirds \
    --data_dir /path/to/waterbirds \
    --model imagenet_resnet50_pretrained \
    --num_epochs 50 \
    --batch_size 32 \
    --lr 0.003 \
    --output_dir outputs/ERM/waterbirds/stage1_seed1
```

**Note**: `--data_dir` should point to your raw dataset directory. Metadata will be auto-generated if missing.

### Stage 2: Extract Embeddings

Extract embeddings from the trained Stage 1 model:

```bash
python scripts/extract_embeddings.py \
    --checkpoint outputs/ERM/waterbirds/stage1_seed1/checkpoint.pt \
    --model imagenet_resnet50_pretrained \
    --num_classes 2 \
    --dataset waterbirds \
    --data_dir /path/to/waterbirds \
    --output_path outputs/ERM/waterbirds/stage1_seed1/embeddings.pt
```

### Stage 3: Train LEIA

Train LEIA on the extracted embeddings:

```bash
python scripts/train.py \
    --method leia \
    --train_embeddings outputs/ERM/waterbirds/stage1_seed1/embeddings.pt \
    --train_labels outputs/ERM/waterbirds/stage1_seed1/embeddings.pt \
    --val_embeddings outputs/ERM/waterbirds/stage1_seed1/embeddings.pt \
    --val_labels outputs/ERM/waterbirds/stage1_seed1/embeddings.pt \
    --test_embeddings outputs/ERM/waterbirds/stage1_seed1/embeddings.pt \
    --test_labels outputs/ERM/waterbirds/stage1_seed1/embeddings.pt \
    --base_weight outputs/ERM/waterbirds/stage1_seed1/embeddings.pt \
    --base_bias outputs/ERM/waterbirds/stage1_seed1/embeddings.pt \
    --spectral_rank 1 \
    --gamma 10.0 \
    --num_epochs 1000 \
    --lr 0.01 \
    --output_dir outputs/LEIA/waterbirds/stage1_seed1
```

**Note**: The embedding extraction script saves everything in a single `.pt` file, so you can use the same path for all arguments.

## Dataset Setup

LEIA automatically generates `metadata.csv` files from raw datasets. Simply point to your dataset directory and metadata will be created automatically.

### CelebA
**Required files:**
- `list_eval_partition.txt` (train/val/test splits)
- `list_attr_celeba.txt` (attributes)
- `img_align_celeba/` directory with images

**Usage:**
```bash
python -m leia.train_stage1 --dataset celeba --data_dir /path/to/celeba
```

### Waterbirds
**Required files:**
- `waterbird_complete95_forest2water2/metadata.csv` (original metadata)

**Usage:**
```bash
python -m leia.train_stage1 --dataset waterbirds --data_dir /path/to/waterbirds
```

### MultiNLI
**Required files:**
- `data/metadata_random.csv` (or `metadata_random.csv` in root)

**Usage:**
```bash
python -m leia.train_stage1 --dataset multinli --data_dir /path/to/multinli
```

### CivilComments
**Required files:**
- `all_data_with_identities.csv` (WILDS format)

**Usage:**
```bash
python -m leia.train_stage1 --dataset civilcomments --data_dir /path/to/civilcomments
```

### Manual Metadata Generation

If you want to generate metadata manually (e.g., before training):

```bash
python -m leia.data.generate_metadata --dataset celeba --data_dir /path/to/celeba
python -m leia.data.generate_metadata --dataset waterbirds --data_dir /path/to/waterbirds
python -m leia.data.generate_metadata --dataset multinli --data_dir /path/to/multinli
python -m leia.data.generate_metadata --dataset civilcomments --data_dir /path/to/civilcomments --granularity coarse
```

The generated `metadata.csv` files will have columns: `id, filename, split, y, a` where:
- `id`: Unique sample ID
- `filename`: Path to data file (relative to dataset directory)
- `split`: 0=train, 1=val, 2=test
- `y`: Class label
- `a`: Attribute/group label

## How LEIA Works

1. **Error-Informed Weights**: Computes weights `μ_i = exp(γ * (1 - p_i))` where `p_i` is the predicted probability of the correct label. Higher weight is assigned to examples with lower confidence.

2. **Error-Weighted Covariance**: Computes `Σ_err = Σ_i μ_i (φ(x_i) - φ̄)(φ(x_i) - φ̄)^T` where `φ(x_i)` are embeddings and `φ̄` is the weighted mean.

3. **Spectral Decomposition**: Extracts top-k eigenvectors `V_k` from `Σ_err`. These eigenvectors span the k-dimensional error subspace.

4. **Low-Rank Adjustment**: Learns adjustment matrix `A ∈ ℝ^(k×C)` such that:
   - `base_logits = W φ(x) + b` (frozen base classifier)
   - `projected = V_k^T φ(x)` (project to error subspace)
   - `adjustment_logits = A^T projected` (learned adjustment)
   - `adjusted_logits = base_logits + adjustment_logits` (final prediction)


[Add your license here]

## Contact

[Add contact information here]
