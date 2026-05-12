# Depression Symptom Severity (DSS) Classification Pipeline

Last Updated: May 12, 2026

Multi-model pipeline for clinical depression severity classification using BERT and Llama language models with class-balanced training on clinical interview transcripts and depression rating scales.

## Pipeline Overview

Multiple training workflows for depression severity prediction:

1. **Train.py** - BERT-based baseline (XLM-RoBERTa)
2. **Train_llm.py** - Llama 3.2-1B with QLoRA fine-tuning
3. **translate.py** - Utilities for text translation and preprocessing

All models train on 3-class depression severity classification:
- Class 0: Mild/None
- Class 1: Moderate
- Class 2: Severe

## Model Architecture

### BERT Model (Train.py)
- Base encoder: FacebookAI/xlm-roberta-base
- Subject-level text chunking for long sequences
- Class-balanced weighted cross-entropy loss
- Optional domain adaptation warmup

### Llama Models (Train_llm.py)
- Base model: Meta Llama-3.2-1B-Instruct
- **QLoRA** (Quantized LoRA) for efficient fine-tuning:
  - 4-bit quantization (NF4 format)
  - LoRA rank: 8, 16, or 32
  - Only ~0.1% parameters trainable
  - Custom classification head (3 classes)
- Class-balanced weighted cross-entropy loss
- Effective-number weighting for imbalanced data

## Setup

### Installation

1. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate dss
```

Alternatively, install via pip:
```bash
pip install -r requirements.txt
```

2. Configure data paths in the training scripts as needed

3. Ensure data files are placed in the `data/` directory (obtain separately)

## File Structure

```
DSS/
├── README.md                         # This file
├── environment.yml                   # Conda environment specification
├── requirements.txt                  # Python package dependencies
├── pipeline/                         # Main training scripts
│   ├── Train.py                      # BERT single-run training
│   ├── Train_llm.py                  # Llama QLoRA single-run               
│   ├── translate.py                  # Text translation utilities
│   ├── learning_results/             # BERT outputs
│   │   └── performance_plots/
│   ├── learning_results_llm/         # Llama single-run outputs
│   │   └── performance_plots/
│   └── learning_results_boundary/    # Boundary-specific results
├── data/                             # ⚠️ NOT INCLUDED IN THIS GIT (CONFIDENTIAL)
│   ├── *.csv                         # Dataset files
│   └── daic/                         # DAIC interview transcripts
├── Paper/                            # Research paper and writeups
└── .gitignore                        # Git ignore rules

**Note:** Data folder is excluded from git due to confidentiality. Access data files through secure channels.
```


## Training Workflows

### Single-Run Training

**BERT Model:**
```bash
python pipeline/Train.py
```

**Llama Model:**
```bash
python pipeline/Train_llm.py
```

- Trains model with configured hyperparameters
- Saves outputs to `pipeline/learning_results/` or `pipeline/learning_results_llm/`
- Evaluates on clinical test data
- On subsequent runs: loads existing checkpoint, skips retraining unless forced

### Text Preprocessing

```bash
python pipeline/translate.py
```

Utilities for text translation and preprocessing of clinical transcripts.

## Data Format

**⚠️ Note:** Data files are not included in this repository due to confidentiality constraints. Data must be obtained separately and placed in the `data/` directory.

Expected clinical dataset format with demographic, temporal, and text features:

Input CSV columns:
- `text` - Clinical notes, interview transcripts, or structured text data
- `labels` - Depression severity class (0, 1, 2) for training data
- `hamd_sum` - Ground truth severity score (HAM-D or equivalent) for evaluation
- `session_id` - Patient/session identifier for grouped analysis

Data sources currently used:
- **DAIC** - Clinician interview transcripts (in `data/daic/` when available)
- **PDCH** - Depression screening data (in `data/` when available)
- **EPIsoDE** - Study-specific clinical data (in `data/` when available)

## Class Balancing

Uses effective-number weighting to handle class imbalance:
$$w_i = \frac{1 - \beta}{1 - \beta^{n_i}}$$
where $\beta = 0.999$, applied as weighted cross-entropy loss.

## Training Configuration

| Setting | Value |
|---------|-------|
| Learning rate | 2e-5 (default) |
| Warmup ratio | 0.03 |
| Batch size | 6-8 |
| Gradient accumulation | 2 steps |
| Epochs | 3-8 |
| Optimizer | AdamW |
| LR scheduler | Cosine annealing |
| Weight decay | 0.01-0.05 |

## Outputs

All models generate performance metrics and visualizations in the respective `learning_results_*` directories:

**Metrics:**
- `*_holdout_overall_metrics.csv` - Accuracy, balanced accuracy, macro F1, Pearson correlation
- `*_holdout_metrics_by_session.csv` - Per-session breakdown

**Visualizations (in `performance_plots/`):**
- Confusion matrices (validation + test)
- Session-grouped scatter plots
- Subject trajectory plots (multi-session subjects only)
- Training curves and loss plots


## Environment

Python 3.10+

Key dependencies:
- transformers (Hugging Face)
- peft (Parameter-Efficient Fine-Tuning)
- bitsandbytes (4-bit quantization)
- torch
- datasets
- scikit-learn

Install:
```bash
pip install torch transformers peft bitsandbytes datasets scikit-learn
```


## Notes

- Absolute paths in scripts; update for your environment
- Grid search is compute-intensive (~324-1728 runs); start with reduced parameter ranges if needed
- QLoRA requires CUDA-capable GPU for 4-bit quantization
- BERT baseline serves as reference; Llama models test larger language model capacity



