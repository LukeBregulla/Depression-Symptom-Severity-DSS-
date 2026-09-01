# New three-stage pipeline

All paths are configured directly near the top of each Python file. No YAML config is used.

Stage 1 uses the RoBERTa NLI checkpoint `cross-encoder/nli-roberta-base`. Stages 2 and 3 use `FacebookAI/xlm-roberta-base`, perform fold-local DAPT, and train a fresh three-class depression head.

## Stages

1. `01_zero_shot_epi.py`
   - Evaluates `cross-encoder/nli-roberta-base` with its NLI head on `data/data_epi.csv`.
   - Scores mild, moderate, and severe depression hypotheses over overlapping transcript chunks.
   - Does not load an old checkpoint or use EPI labels for fitting.
   - Writes metrics, subject predictions, and a confusion matrix to `results/01_zero_shot_epi`.

2. `02_train_test_epi.py`
   - Runs fresh fold-local DAPT and five-fold cross-validation on EPIsoDE only.
   - Has no separate test file: each validation fold is the held-out test partition for that fold.
   - Saves subject-aligned five-fold out-of-fold predictions and evaluates them as the within-EPI validation result.
   - The best held-out fold confusion matrix, balanced accuracy, and macro F1 are the headline diagnostic; all-fold scores and full OOF predictions remain available as diagnostics.
   - Writes the validation metrics CSV, confusion matrix, and OOF predictions to `results/02_epi`.
   - Also writes fold-level class support and the majority-class baseline for transparent reporting.
   - Does not perform a final all-data refit because only within-EPI performance is required.
   - Because EPI contains only one mild subject, this result should be reported as exploratory within-cohort validation rather than a stable three-class estimate.

3. `03_train_daic_pdch.py`, then `03_test_epi.py`
   - Trains on `data/data_pdch.csv` and `data/data_daic.csv`.
   - Tests on `data/data_epi.csv`.
   - Reports five-fold OOF training metrics and the plain EPI cross-cohort test result in `results/03_pdch_daic_epi/test`.
   - No supervised EPI adaptation is attempted because EPI has only one mild/class-0 subject, which cannot support valid five-fold stratification.
   - The best fold's classifier checkpoint and matching DAPT encoder are saved and loaded by `03_test_epi.py`.

## Commands

Run from the DSS project root:

```bash
python pipeline/01_zero_shot_epi.py
python pipeline/02_train_test_epi.py
python pipeline/03_train_daic_pdch.py
python pipeline/03_test_epi.py
```

Each evaluation stage writes an overall metrics CSV and a confusion-matrix PNG under its configured result directory. The Stage 3 training script also writes the final model checkpoint needed by the EPI test script.
