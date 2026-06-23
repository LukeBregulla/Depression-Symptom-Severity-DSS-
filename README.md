# Depression Symptom Severity (DSS) Classification Pipeline


Transformer-based depression severity classification from clinical interview transcripts using XLM-RoBERTa 

---

## Overview

The DSS pipeline implements an end-to-end workflow for training and evaluating depression severity classifiers on clinical text data:

1. **Training** (`pipeline/Train.py`) – Domain adaptation + 3-class classifier training with validation evaluation
2. **Testing** (`pipeline/Test.py`) – Flexible holdout/test set evaluation with configurable label subsets
3. **Configuration** (`pipeline/config.yaml`) – YAML-based config settings



## Data Format

### Input CSV Structure

**Required columns:**
- `text` (str) – Clinical interview transcript or document text
- `hamd_sum` (int) – HAM-D depression screening score for labels
- `session_id` (str, format: `SUBJECT-SESSION`) – Subject identifier for permutation testing

**Optional columns:**
- Additional demographic or clinical metadata (ignored by pipeline)


### HAM-D to Depression Classes

Labels are derived from HAM-D (Hamilton Depression Rating Scale):

| HAM-D Score | Class | Label |
|------------|-------|-------|
| ≤ 7 | None/Mild | 0 |
| 8–23 | Moderate | 1 |
| ≥ 24 | Severe | 2 |


## File Structure

```
DSS/
├── README.md
├── environment.yml
├── requirements.txt
├── pipeline/
│   ├── Train.py                    # Training entry point
│   ├── Test.py                     # Evaluation entry point
│   ├── config.yaml                 # Configuration file
│   └── translate.py                # Text utilities
├── data/                           # ⚠️ NOT INCLUDED

├── results/
│   ├── Validation/                 # Training validation outputs
│   │   ├── domain_adapted_bert/
│   │   ├── best_model/
│   │   └── validation_overall_metrics.csv
│   ├── test1/                      # Test configuration 1 output
│   ├── test2/                      # Test configuration 2 output
│   └── Validation/
└── .gitignore
```

---

## Training Configuration

Default hyperparameters in `Train.py`:

| Parameter | Value |
|-----------|-------|
| Base Model | FacebookAI/xlm-roberta-base |
| Batch Size | 8 |
| Learning Rate | 3e-5 |
| Warmup Ratio | 0.08 |
| Epochs | 8 |
| Optimizer | AdamW |
| Weight Decay | 0.09 |
| Dropout | 0.3 |
| LR Scheduler | Cosine |
| Gradient Accumulation steps | 1 |
| Frozen Layers | 4 |

---

## Outputs

### Training Outputs (`results/Validation/`)

- `domain_adapted_bert/` – Domain-adapted encoder weights and config
- `best_model/` – Best classifier checkpoint
- `validation_overall_metrics.csv` – Final validation performance
- `validation_subject_predictions.csv` – Per-subject predictions
- `performance_plots/` – Confusion matrix, ROC curves, class distribution plots

### Evaluation Outputs (e.g., `results/2_class/`)

- `holdout_overall_metrics.csv` – Summary metrics
- `holdout_precision_recall_roc_auc.csv` – Per-class details
- `holdout_subject_predictions.csv` – Predictions with row IDs
- `performance_plots/confusion_matrix.png` – Confusion matrix heatmap
- `performance_plots/roc_curves.png` – ROC curves (one per class)
- `performance_plots/class_distribution.png` – Test set class proportions

---


## Quick Start

### 1. Environment Setup

```bash
conda env create -f environment.yml
conda activate env
```

Or with pip:
```bash
pip install -r requirements.txt
```

### 2. Training (Validation Set)

```bash
cd pipeline
python Train.py
```

Outputs to `results/Validation/`:
- Domain-adapted encoder
- Best classifier weights
- Validation metrics and plots

### 3. Holdout Evaluation

Edit `pipeline/config.yaml` to specify test data and evaluation settings:

```bash
cd pipeline
python Test.py
```

Outputs to `results/test/`:
- Test metrics 
- Performance plots

---

## Configuration

All holdout evaluation settings are in `pipeline/config.yaml`. Edit before running `Test.py`:

```yaml
test_data:
  file: "data/test.csv"
  sep: "\t" #specify seperation type
  encoding: "utf-8"

output:
  directory: "results/test"
  name: "test"

model:
  base_model: "FacebookAI/xlm-roberta-base"
  domain_adapted_bert_dir: "results/Validation/domain_adapted_bert" #saved from training
  best_model_dir: "results/Validation/best_model" #saved from training

evaluation:
  labels: 3  # 1 = [0] only, 2 = [1,2] (moderate+severe), 3 = [0,1,2] (all)
```




