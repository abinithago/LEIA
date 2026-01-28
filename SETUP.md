# LEIA Setup Guide

This guide will help you set up LEIA for reproduction.

## Quick Start

1. **Create Conda Environment**:
```bash
conda env create -f environment.yml
conda activate leia_env
```

2. **Update Dataset Paths**: Edit `leia/config.py` and update `DATASET_PATHS` dictionary with your dataset locations.

3. **Run Stage 1 Training**:
```bash
python -m leia.train_stage1 \
    --dataset waterbirds \
    --data_dir /path/to/your/waterbirds \
    --model imagenet_resnet50_pretrained \
    --output_dir outputs/ERM/waterbirds/stage1_seed1
```

## TODOs in the Codebase

### 1. Dataset Paths (`leia/config.py`)
- [ ] Update `DATASET_PATHS` dictionary with your dataset locations
- [ ] Ensure each dataset directory contains the required metadata files

### 2. Dataset Loaders (`leia/data/datasets.py`)
- [ ] Verify metadata file paths match your dataset structure
- [ ] Update image path resolution if your dataset uses a different structure

### 3. Training Scripts
- [ ] Update default paths in `train_stage1.py`, `extract_embeddings.py`, and `train.py` if needed
- [ ] Adjust hyperparameters based on your dataset and hardware

## Dataset Requirements

Each dataset should have:
- **CelebA/Waterbirds**: `metadata.csv` with columns: `img_filename`, `y`, `place`, `split`
- **CivilComments**: WILDS-compatible format
- **MultiNLI**: `metadata_random.csv` and pre-computed BERT features

## File Structure

```
leia/
├── leia/                    # Main package
│   ├── data/                # Dataset loaders (with TODOs for paths)
│   ├── losses/              # LEIA loss functions
│   ├── models/              # LEIAModel and architectures
│   ├── optimizers/          # Optimizer factories
│   ├── utils/               # Metrics, logging, training utilities
│   ├── config.py            # Configuration (TODO: update paths)
│   ├── train_stage1.py      # Stage 1 training
│   ├── extract_embeddings.py # Embedding extraction
│   └── train.py             # Stage 2 LEIA training
├── environment.yml          # Conda environment file
├── README.md                # Main documentation
└── SETUP.md                 # This file
```

## Notes

- All SEL references have been renamed to LEIA
- No Jupyter notebooks are included (as requested)
- All dataset paths have TODOs for easy identification
- The codebase is self-contained and ready for reproduction
