# Depression Symptom Severity (DSS) Classification Pipeline

**Version:** 2.0  
**Last Updated:** June 2026

Multilingual transformer-based depression severity classification from clinical interview transcripts using XLM-RoBERTa with domain adaptation and subject-wise statistical testing.

---

## Overview

The DSS pipeline implements an end-to-end workflow for training and evaluating depression severity classifiers on clinical text data:

1. **Training** (`pipeline/Train.py`) – Domain adaptation + 3-class classifier training with validation evaluation
2. **Evaluation** (`pipeline/Test.py`) – Flexible holdout/test set evaluation with configurable label subsets
3. **Configuration** (`pipeline/config.yaml`) – YAML-based settings for model paths, data, and evaluation

The pipeline supports:
- **Flexible class evaluation:** 3-class (none/moderate/severe), 2-class (moderate/severe), or 1-class subsets
- **Subject-wise permutation testing:** Statistical significance with autocorrelation preservation
- **Per-class and overall metrics:** Balanced accuracy, macro-F1, ROC-AUC, precision, recall
- **Multiple data sources:** EPI (V1/V12), DAIC, PDCH

---

## Quick Start

### 1. Environment Setup

```bash
conda env create -f environment.yml
conda activate py1
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

---

## Configuration

All holdout evaluation settings are in `pipeline/config.yaml`. Edit before running `Test.py`:

```yaml
test_data:
  file: "data/data_epi_en_V12.csv"
  sep: "\t"
  encoding: "utf-8"

output:
  directory: "results/2_class"
  name: "holdout"

model:
  base_model: "FacebookAI/xlm-roberta-base"
  domain_adapted_bert_dir: "results/Validation/domain_adapted_bert"
  best_model_dir: "results/Validation/best_model"

evaluation:
  label_source: "hamd_three_class"
  labels: 2  # 1 = [0] only, 2 = [1,2] (moderate+severe), 3 = [0,1,2] (all)
  seed: 42
```

### Label Configuration

The `evaluation.labels` parameter controls which depression classes are evaluated:

| Value | Classes | Interpretation |
|-------|---------|-----------------|
| **1** | [0] | None/mild only |
| **2** | [1, 2] | Moderate + Severe |
| **3** | [0, 1, 2] | All classes |

All metrics (balanced accuracy, F1, ROC-AUC, permutation test) dynamically adjust to the configured subset.

### Quick Switches

Uncomment in `config.yaml` to switch configurations:

```yaml
# Test V1 instead of V12:
# test_data:
#   file: "data/data_epi_en_V1.csv"
# output:
#   directory: "results/V1"
#   name: "holdout_V1"

# Test 2-class subset:
# evaluation:
#   labels: 2
# output:
#   directory: "results/2_class"
#   name: "holdout_2class"
```

---

## Model Architecture

### Encoder
- **Base:** FacebookAI/xlm-roberta-base (XLM-RoBERTa)
- **Language support:** 100+ languages
- **Sequence handling:** Subject-level text chunking with overlapping windows
  - Max chunk length: 512 tokens
  - Chunk stride: 256 tokens
  - Aggregation: Mean pooling of chunk representations

### Classifier Head
- **Input:** Subject-level mean-pooled embedding (768-dim)
- **Architecture:** Linear layer → 3-class softmax
- **Dropout:** 0.3 (during training)

### Class Balancing
Effective-number class weighting to handle imbalanced data:
$$w_i = \frac{1 - \beta}{1 - \beta^{n_i}}$$

where $\beta = 0.999$ and $n_i$ is the number of samples in class $i$. Applied as weighted cross-entropy loss.

---

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

### Data Sources

- **EPI V1 / V12** – EPIsoDE study data (German language clinical interviews)
- **DAIC-WOZ** – DAIC Wizard-of-Oz interviews (English, partial dataset)
- **PDCH** – German depression screening cohort data

⚠️ **Data Confidentiality:** Data files are excluded from this repository. Obtain via institutional data access agreements.

---

## Evaluation Metrics

### Overall Metrics

**`overall_metrics.csv`** contains:

| Metric | Definition |
|--------|-----------|
| `n` | Number of test samples |
| `accuracy` | (TP + TN) / Total |
| `balanced_accuracy` | Avg. per-class recall |
| `macro_f1` | Unweighted avg. F1 across configured classes |
| `roc_auc_ovr_macro` | Macro-averaged ROC-AUC (one-vs-rest) |
| `p_value_right_tailed` | Permutation test p-value |
| `null_macro_f1_mean` | Null distribution mean (from 10,000 permutations) |

### Per-Class Metrics

**`precision_recall_roc_auc.csv`** contains for each configured class:

| Metric | Definition |
|--------|-----------|
| `class_id` | Depression class (0, 1, or 2) |
| `precision` | TP / (TP + FP) |
| `recall` | TP / (TP + FN) |
| `roc_auc` | One-vs-rest area under ROC curve |

### Permutation Testing

Subject-wise permutation test for statistical significance:

1. **Null hypothesis:** Random assignment of subject depression classes yields equivalent macro-F1
2. **Procedure:** 
   - Randomly permute subject-level depression labels (10,000 iterations)
   - Broadcast permuted label to all sessions/chunks of each subject
   - Compute macro-F1 on the configured label space
   - Right-tailed p-value: P(null_F1 ≥ observed_F1)

3. **Rationale:** Preserves within-subject autocorrelation (multiple sessions per subject are dependent)

4. **Output:** `p_value_right_tailed` and `null_macro_f1_mean` in metrics CSV

---

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
├── data/                           # ⚠️ NOT INCLUDED (confidential)
│   ├── data_epi_en_V12.csv
│   ├── data_epi_en_V1.csv
│   └── data_daic_pdch_3class_texts.csv
├── results/
│   ├── Validation/                 # Training validation outputs
│   │   ├── domain_adapted_bert/
│   │   ├── best_model/
│   │   └── validation_overall_metrics.csv
│   ├── V1/                         # EPI V1 holdout outputs
│   ├── V12/                        # EPI V12 holdout outputs
│   ├── 2_class/                    # 2-class evaluation outputs
│   └── Validation/
├── Paper/                          # Manuscript/publication materials
├── Endnote/                        # Reference library
└── .gitignore
```

---

## Training Configuration

Default hyperparameters in `Train.py`:

| Parameter | Value |
|-----------|-------|
| Base Model | FacebookAI/xlm-roberta-base |
| Batch Size | 6–8 |
| Learning Rate | 2e-5 |
| Warmup Ratio | 0.03 |
| Epochs | 3–8 |
| Optimizer | AdamW |
| Weight Decay | 0.01–0.05 |
| Dropout | 0.3 |
| LR Scheduler | Cosine annealing |
| Gradient Accumulation | 2 steps |

---

## Outputs

### Training Outputs (`results/Validation/`)

- `domain_adapted_bert/` – Domain-adapted encoder weights and config
- `best_model/` – Best classifier checkpoint (from validation F1)
- `validation_overall_metrics.csv` – Final validation performance
- `validation_subject_predictions_V*.csv` – Per-subject predictions
- `performance_plots/` – Confusion matrix, ROC curves, class distribution plots

### Evaluation Outputs (e.g., `results/2_class/`)

- `holdout_overall_metrics.csv` – Summary metrics
- `holdout_precision_recall_roc_auc.csv` – Per-class details
- `holdout_subject_predictions.csv` – Predictions with row IDs
- `holdout_metrics_by_session.csv` – Session-level breakdowns (if available)
- `performance_plots/confusion_matrix.png` – Confusion matrix heatmap
- `performance_plots/roc_curves.png` – ROC curves (one per class)
- `performance_plots/class_distribution.png` – Test set class proportions

---

## Dependencies

**Core:**
- Python ≥ 3.10
- torch ≥ 2.0
- transformers ≥ 4.30
- datasets ≥ 2.10
- scikit-learn ≥ 1.3

**Optional:**
- peft (for parameter-efficient fine-tuning)
- bitsandbytes (for 4-bit quantization)
- matplotlib (for plotting)
- PyYAML (for config loading)

Install from `environment.yml` (recommended) or `requirements.txt`.

---

## Notes

- **Validation outputs** are fixed to `results/Validation/` for reproducibility
- **Holdout outputs** are configured via `pipeline/config.yaml`
- **Subject-wise grouping** is extracted from `session_id` column (format: `SUBJECT-SESSION`)
- **Permutation test** respects the configured label subset, adjusting both observed and null distributions
- **Metric computation** adapts dynamically to the number of classes in the evaluation set

---

## References

- **XLM-RoBERTa:** Conneau et al. (2020). "Unsupervised Cross-lingual Representation Learning at Scale." ACL.
- **HAM-D Scale:** Hamilton, M. (1960). "A rating scale for depression." Journal of Neurology, Neurosurgery & Psychiatry.
- **Effective Number of Samples:** Cui et al. (2019). "Class-Balanced Loss Based on Effective Number of Samples." CVPR.

---

## License & Contact

Data access and usage policies available upon request. For questions, contact the project maintainers.



