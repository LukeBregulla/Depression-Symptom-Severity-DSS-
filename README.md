# HAM-D Severity Prediction Pipeline

Last Updated: April 30, 2026

Multi-model pipeline for clinical depression severity classification using BERT and Llama language models with class-balanced training.

## Pipeline Overview

Three parallel training workflows:

1. **Train.py** - BERT-based baseline (XLM-RoBERTa)
2. **Train_llm.py** - Llama 3.2-1B with QLoRA fine-tuning


All models train 3-class depression severity classification:
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



## File Structure

```
predict_hamd/
├── Train.py                          # BERT single-run training
├── Train_llm.py                      # Llama QLoRA single-run               
├── learning_results/                 # BERT outputs
│   ├── best_model/                   # Best BERT checkpoint (not included in this git)
│   └── performance_plots/
├── learning_results_llm/             # Llama single-run outputs
│   ├── best_model/                   # Llama checkpoint ((not included in this git))
│   └── performance_plots/


## Training Workflows

### Single-Run Training
```bash
python predict_hamd/Train.py/*_llm.py
```
- Trains BERT/ Llama model with fixed hyperparameters
- Saves checkpoint to `learning_results/*_llm/`
- Evaluates on held-out clinical data
- On second run: loads existing checkpoint, skips training

## Data Format

Clinical dataset with demographic, temporal, and text features. Input CSV columns:
- `text` - Clinical notes or transcripts
- `labels` - Depression severity class (0, 1, 2) for training data
- `hamd_sum` - Ground truth severity score for evaluation
- `session_id` - Patient/session identifier for grouped analysis

**Note:** Data paths are environment-specific; update paths in training scripts as needed.


**Checkpoints:**
- `best_model/` - Full model weights + LoRA adapters (for Llama)

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

All models generate:

**Metrics:**
- `..._holdout_overall_metrics.csv` - Accuracy, balanced accuracy, macro F1, Pearson correlation
- `..._holdout_metrics_by_session.csv` - Per-session breakdown

**Visualizations:**
- Confusion matrices (validation + test)
- Session-grouped scatter plots
- Subject trajectory plots (multi-session subjects only)


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



