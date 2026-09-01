import os
import gc
import csv
import re
import shutil
import random
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, auc, average_precision_score, precision_recall_curve, balanced_accuracy_score
import matplotlib.pyplot as plt

# Permutation testing utilities
def macro_f1_present_labels(y_true, y_pred, label_space=None):
    """Compute macro F1 only on labels present in label_space."""
    if label_space is None:
        label_space = np.unique(y_true)
    label_space = np.asarray(label_space, dtype=np.int64)
    return float(f1_score(y_true, y_pred, labels=label_space, average="macro", zero_division=0))

def participant_key(sample_id):
    """Collapse session-level ids (ZI002-1, PDCH_001A) to one key per participant."""
    return re.sub(r"(-\d+|(?<=\d)[A-Za-z])$", "", str(sample_id))


def permutation_participant_wise(y_true, y_pred, participant_ids, n_permutations=10000, seed=42, label_space=None):
    """Permutation test for macro-F1 vs chance, permuting whole participants.

    Sessions of the same participant are not independent, so labels are exchanged between
    participants (within equal block sizes) rather than between individual sessions.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    keys = np.asarray([participant_key(p) for p in participant_ids])
    if keys.size != y_true.size:
        raise ValueError("participant_ids length must match y_true length")

    label_space = np.unique(y_true) if label_space is None else np.asarray(label_space, dtype=np.int64)
    observed = macro_f1_present_labels(y_true, y_pred, label_space=label_space)

    blocks = [np.where(keys == p)[0] for p in np.unique(keys)]
    size_groups = {}
    for block in blocks:
        size_groups.setdefault(block.size, []).append(block)

    rng = np.random.default_rng(seed)
    perm_scores = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        y_perm = np.empty_like(y_true)
        for group in size_groups.values():
            order = rng.permutation(len(group))
            for target_idx, source_idx in enumerate(order):
                y_perm[group[target_idx]] = y_true[group[source_idx]]
        perm_scores[i] = macro_f1_present_labels(y_perm, y_pred, label_space=label_space)

    return {
        "observed_macro_f1": observed,
        "null_macro_f1_mean": float(np.mean(perm_scores)),
        "perm_p_value": float((np.sum(perm_scores >= observed) + 1) / (n_permutations + 1)),
        "n_permutations": int(n_permutations),
        "n_participants": int(len(blocks)),
        "n_sessions": int(y_true.size),
    }


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reseed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# -----------------------------------------------------------------------------
# Paths and config
# -----------------------------------------------------------------------------

PROJECT_ROOT = "/zi/home/luke.bregulla/Desktop/DSS"

PDCH_FILE = os.path.join(PROJECT_ROOT, "data/data_pdch.csv")
DAIC_FILE = os.path.join(PROJECT_ROOT, "data/data_daic.csv")
# EPI_FILE moved to 04_predict_epi.py

RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results/03_daic")
OUTPUT_DIR = RESULTS_ROOT
LOGGING_DIR = os.path.join(OUTPUT_DIR, "logs")
BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model")
SUBJECT_PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "subject_predictions.csv")
SEVERE_METRICS_CSV = os.path.join(OUTPUT_DIR, "severe_discrimination_metrics.csv")
VALIDATION_METRICS_CSV = os.path.join(OUTPUT_DIR, "validation_metrics.csv")
PLOT_DIR = os.path.join(OUTPUT_DIR, "performance_plots")

os.environ["TENSORBOARD_LOGGING_DIR"] = LOGGING_DIR

BASE_MODEL = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Use 8 subjects per batch for stable gradient dynamics and better batch statistics (critical for performance)
SUBJECT_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
CHUNK_MICRO_BATCH_SIZE = 8
MAX_LEN = 512
STRIDE = 256
USE_GRADIENT_CHECKPOINTING = True
FREEZE_BOTTOM_LAYERS = 4
USE_FP16 = True  # CRITICAL: Must be True for optimal training (disabling causes significant performance degradation)

NUM_CLASSES = 2
CLASS_NAMES = ["mild_moderate", "severe"]
LABEL_SPACE = [0, 1]

CLASS_BALANCE_BETA = 0.999
USE_CLASS_WEIGHTS = True

MAX_EPOCHS = 20
LEARNING_RATE = 3e-5

# Hardcoded hyperparameters (best from prior grid search)
EFN_WEIGHT_POWER = 1.0
WEIGHT_DECAY = 0.01
DROPOUT_RATE = 0.3
FINAL_N_FOLDS = 5


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def detect_separator(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
    return "\t" if header.count("\t") > header.count(",") else ","


def load_csv_dataset(path: str, split: str):
    sep = detect_separator(path)
    return load_dataset("csv", data_files={split: path}, sep=sep, encoding="utf-8")[split]



def build_class_weights(target_classes: np.ndarray, power: float) -> tuple[np.ndarray, np.ndarray]:
    cls = np.asarray(target_classes, dtype=np.int64).reshape(-1)
    counts = np.bincount(cls, minlength=NUM_CLASSES).astype(np.float64)
    if not USE_CLASS_WEIGHTS or power <= 0.0:
        return np.ones(NUM_CLASSES, dtype=np.float32), counts.astype(np.int64)

    effective_num = np.zeros_like(counts)
    nonzero = counts > 0
    effective_num[nonzero] = (1.0 - np.power(CLASS_BALANCE_BETA, counts[nonzero])) / (1.0 - CLASS_BALANCE_BETA)

    weights = np.zeros_like(counts)
    weights[nonzero] = 1.0 / np.maximum(effective_num[nonzero], 1e-8)
    if np.any(nonzero):
        weights[nonzero] = weights[nonzero] / np.mean(weights[nonzero])
        weights[nonzero] = np.power(weights[nonzero], power)
        weights[nonzero] = weights[nonzero] / np.mean(weights[nonzero])
    return weights.astype(np.float32), counts.astype(np.int64)


def write_rows_csv(output_path: str, fieldnames: List[str], rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_single_row_csv(output_path: str, row: Dict) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def save_roc_and_pr_curves(y_true, prob_severe, output_path):
    """Save severe-positive ROC and precision-recall curves from pooled OOF predictions."""
    y_true = np.asarray(y_true, dtype=np.int64)
    prob_severe = np.asarray(prob_severe, dtype=np.float64)
    severe_true = (y_true == 1).astype(np.int64)
    if np.unique(severe_true).size < 2:
        return np.nan, np.nan
    fpr, tpr, _ = roc_curve(severe_true, prob_severe)
    roc_auc = float(auc(fpr, tpr))
    precision, recall, _ = precision_recall_curve(severe_true, prob_severe)
    average_precision = float(average_precision_score(severe_true, prob_severe))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].plot(fpr, tpr, color="#1F77B4", linewidth=2.0, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2)
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC: Severe vs Mild/Moderate")
    axes[0].legend(loc="lower right", frameon=False)
    axes[0].grid(alpha=0.2)
    axes[1].plot(recall, precision, color="#FF7F0E", linewidth=2.0, label=f"Average precision = {average_precision:.3f}")
    axes[1].axhline(severe_true.mean(), linestyle="--", color="gray", linewidth=1.2, label="Prevalence")
    axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision", title="Precision-Recall: Severe vs Mild/Moderate")
    axes[1].legend(loc="best", frameon=False)
    axes[1].grid(alpha=0.2)
    fig.suptitle("Pooled Out-of-Fold Evaluation", y=1.03)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return roc_auc, average_precision


def save_class_distribution_plot(distribution_map, output_path):
    """Save class distribution plot."""
    labels = ["mild_moderate", "severe"]
    dataset_items = list(distribution_map.items())

    if len(dataset_items) != 2:
        # Fallback to single-axis layout if caller passes a different number of datasets.
        x = np.arange(len(labels), dtype=float)
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        for i, (name, counts) in enumerate(dataset_items):
            counts = np.asarray(counts, dtype=int)
            bars = ax.bar(x + i * 0.25, counts, width=0.25, label=name, alpha=0.9)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.5, f"{int(h)}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x + 0.25 * max(len(dataset_items) - 1, 0) / 2.0)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Class")
        ax.set_ylabel("Number of cases")
        ax.set_title("Class Distribution")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
        for ax, (name, counts) in zip(axes, dataset_items):
            counts = np.asarray(counts, dtype=int)
            x = np.arange(len(labels), dtype=float)
            bars = ax.bar(x, counts, width=0.62, color="#4C78A8", alpha=0.9)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.5, f"{int(h)}", ha="center", va="bottom", fontsize=8)

            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_xlabel("Class")
            ax.set_title(name)
            ax.grid(axis="y", alpha=0.25)

        axes[0].set_ylabel("Number of cases")
        fig.suptitle("Class Distribution: DAIC (left) vs PDCH (right)")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def safe_float(v):
    try:
        if pd.isna(v):
            return np.nan
        return float(v)
    except Exception:
        return np.nan



# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------

daic_raw = load_csv_dataset(DAIC_FILE, "train")

if "labels" not in daic_raw.column_names:
    raise ValueError(f"Missing 'labels' column in DAIC: {DAIC_FILE}")


def prep_source_dataset(raw_ds, source_name: str):
    labels = np.asarray(raw_ds["labels"], dtype=np.int64).reshape(-1)
    if sorted(np.unique(labels).tolist()) != LABEL_SPACE:
        raise ValueError(f"{source_name} labels must already be [0, 1]")
    if "continuous_score" not in raw_ds.column_names:
        raise ValueError(f"{source_name} dataset is missing 'continuous_score'")
    ids = [str(x) for x in raw_ds["id"]]
    texts = [str(x) for x in raw_ds["text"]]

    rows = []
    for i in range(len(ids)):
        sid = ids[i]
        score = safe_float(raw_ds["continuous_score"][i])
        rows.append(
            {
                "id": sid,
                "text": texts[i],
                "labels": int(labels[i]),
                "source": source_name,
                "continuous_score": score,
            }
        )
    return rows


daic_rows = prep_source_dataset(daic_raw, "DAIC")
train_rows = daic_rows  # Train on DAIC only

os.makedirs(OUTPUT_DIR, exist_ok=True)
tmp_train_csv = os.path.join(OUTPUT_DIR, "_tmp_train.csv")
pd.DataFrame(train_rows).to_csv(tmp_train_csv, index=False)
train_dataset = DatasetDict({"train": load_csv_dataset(tmp_train_csv, "train")})
os.remove(tmp_train_csv)

train_dataset["train"] = train_dataset["train"].map(
    lambda x: {
        "id": str(x["id"]),
        "text": str(x["text"]),
        "labels": int(x["labels"]),
        "source": str(x["source"]),
        "continuous_score": safe_float(x["continuous_score"]),
    },
    remove_columns=train_dataset["train"].column_names,
)
train_dataset["train"] = train_dataset["train"].select_columns(["id", "text", "labels", "source", "continuous_score"])

train_classes = np.asarray(train_dataset["train"]["labels"], dtype=np.int64)
train_participants = np.asarray([participant_key(sample_id) for sample_id in train_dataset["train"]["id"]], dtype=object)
print(f"Training on DAIC only | n={len(train_classes)} | counts={np.bincount(train_classes, minlength=NUM_CLASSES).tolist()}")


# -----------------------------------------------------------------------------
# Model and collator
# -----------------------------------------------------------------------------

class CustomBERTModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        class_weights: np.ndarray,
        chunk_micro_batch_size: int,
        freeze_bottom_layers: int,
        dropout_rate: float,
        use_gradient_checkpointing: bool,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name, torch_dtype=torch.float32)
        self.config = self.encoder.config
        self.config.num_labels = NUM_CLASSES
        self.chunk_micro_batch_size = chunk_micro_batch_size
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, NUM_CLASSES)
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))

        if freeze_bottom_layers > 0:
            for p in self.encoder.embeddings.parameters():
                p.requires_grad = False
            if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
                total_layers = len(self.encoder.encoder.layer)
                n_freeze = min(int(freeze_bottom_layers), total_layers)
                for layer_idx in range(n_freeze):
                    for p in self.encoder.encoder.layer[layer_idx].parameters():
                        p.requires_grad = False

        if use_gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def _encode_chunks_sequential(self, input_ids, attention_mask):
        cls_parts = []
        total_chunks = input_ids.size(0)
        for start in range(0, total_chunks, self.chunk_micro_batch_size):
            end = min(start + self.chunk_micro_batch_size, total_chunks)
            out = self.encoder(input_ids=input_ids[start:end], attention_mask=attention_mask[start:end])
            cls_parts.append(self.dropout(out.last_hidden_state[:, 0, :]))
        return torch.cat(cls_parts, dim=0)

    def forward(self, input_ids=None, attention_mask=None, labels=None, subject_chunk_counts=None):
        if subject_chunk_counts is not None:
            logits_list = []
            subject_labels_list = []
            chunk_idx = 0
            for subject_idx, n_chunks in enumerate(subject_chunk_counts):
                subj_input_ids = input_ids[chunk_idx: chunk_idx + n_chunks]
                subj_attention = attention_mask[chunk_idx: chunk_idx + n_chunks]
                cls_emb = self._encode_chunks_sequential(subj_input_ids, subj_attention)
                pooled = torch.mean(cls_emb, dim=0)
                pooled = pooled.to(self.classifier.weight.dtype)
                logits_list.append(self.classifier(pooled))
                if labels is not None:
                    subject_labels_list.append(labels[subject_idx])
                chunk_idx += n_chunks

            subject_logits = torch.stack(logits_list)
            if labels is not None:
                subject_labels = torch.stack(subject_labels_list).long()
                loss = torch.nn.functional.cross_entropy(subject_logits, subject_labels, weight=self.class_weights)
                return SequenceClassifierOutput(loss=loss, logits=subject_logits)
            return SequenceClassifierOutput(logits=subject_logits)

        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        cls = cls.to(self.classifier.weight.dtype)
        logits = self.classifier(cls)
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels.long(), weight=self.class_weights)
            return SequenceClassifierOutput(loss=loss, logits=logits)
        return SequenceClassifierOutput(logits=logits)


class SubjectChunkingCollator:
    def __init__(self, tokenizer_, max_len=MAX_LEN, stride=STRIDE):
        self.tokenizer = tokenizer_
        self.max_len = max_len
        self.stride = stride

    def __call__(self, batch):
        all_input_ids = []
        all_attention_masks = []
        subject_labels = []
        subject_chunk_counts = []
        has_targets = "target_class" in batch[0]

        for ex in batch:
            text = str(ex["text"]).lower()
            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_len,
                stride=self.stride,
                return_overflowing_tokens=True,
            )
            all_input_ids.extend(enc["input_ids"])
            all_attention_masks.extend(enc["attention_mask"])
            if has_targets:
                subject_labels.append(int(ex["target_class"]))
            subject_chunk_counts.append(len(enc["input_ids"]))

        padded = self.tokenizer.pad(
            {"input_ids": all_input_ids, "attention_mask": all_attention_masks},
            padding=True,
            return_tensors="pt",
        )
        out = {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "subject_chunk_counts": subject_chunk_counts,
        }
        if has_targets:
            out["labels"] = torch.tensor(subject_labels, dtype=torch.long)
        return out


# -----------------------------------------------------------------------------
# Training/eval helpers
# -----------------------------------------------------------------------------


def extract_pred_and_prob(predictions):
    preds = predictions[0] if isinstance(predictions, (tuple, list)) else predictions
    logits = np.asarray(preds, dtype=float)
    if logits.ndim == 1:
        logits = logits.reshape(-1, NUM_CLASSES)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
    pred_classes = np.argmax(logits, axis=1).astype(np.int64)
    return pred_classes, probs


def best_epoch_from_log(log_history):
    best_score, best_epoch = None, None
    for rec in log_history:
        if "eval_macro_f1" in rec and (best_score is None or rec["eval_macro_f1"] > best_score):
            best_score = rec["eval_macro_f1"]
            best_epoch = rec.get("epoch")
    return int(round(best_epoch)) if best_epoch else None


def compute_metrics(eval_pred):
    predictions, true_classes = eval_pred
    pred_classes, _ = extract_pred_and_prob(predictions)
    true_classes = np.asarray(true_classes, dtype=np.int64).reshape(-1)
    acc = float(np.mean(pred_classes == true_classes)) if true_classes.size else 0.0
    bal_acc = balanced_accuracy_score(true_classes, pred_classes) if true_classes.size else 0.0
    macro = float(f1_score(true_classes, pred_classes, labels=LABEL_SPACE, average="macro", zero_division=0)) if true_classes.size else 0.0
    return {"class_accuracy": acc, "balanced_accuracy": bal_acc, "macro_f1": macro}


def run_cv_for_config(efn_power: float, weight_decay: float, dropout_rate: float, config_dir: str, n_folds: int):
    if n_folds < 2 or n_folds > 5:
        raise ValueError(f"n_folds must be between 2 and 5, got {n_folds}")
    split = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_splits = list(split.split(np.zeros(len(train_dataset["train"])), train_classes, groups=train_participants))
    fold_scores = []
    fold_best_epochs = []
    best_fold_score = -np.inf

    oof_true = np.full(len(train_dataset["train"]), -1, dtype=np.int64)
    oof_pred = np.full(len(train_dataset["train"]), -1, dtype=np.int64)
    oof_prob = np.full(len(train_dataset["train"]), np.nan, dtype=np.float64)

    fold_rows = []

    for fold_idx, (tr_idx, va_idx) in enumerate(fold_splits, start=1):
        reseed()
        print(f"\n--- Fold {fold_idx}/{n_folds} ---")

        fold_ds = DatasetDict(
            {
                "train": train_dataset["train"].select(tr_idx.tolist()),
                "validation": train_dataset["train"].select(va_idx.tolist()),
            }
        )
        fold_ds["train"] = fold_ds["train"].map(lambda x: {**x, "target_class": int(x["labels"])})
        fold_ds["validation"] = fold_ds["validation"].map(lambda x: {**x, "target_class": int(x["labels"])})

        fold_train_cls = np.asarray(fold_ds["train"]["target_class"], dtype=np.int64)
        class_weights_np, _ = build_class_weights(fold_train_cls, power=efn_power)

        fold_output_dir = os.path.join(config_dir, f"fold_{fold_idx}")
        shutil.rmtree(fold_output_dir, ignore_errors=True)

        args = TrainingArguments(
            output_dir=fold_output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=LEARNING_RATE,
            warmup_ratio=0.08,
            per_device_train_batch_size=SUBJECT_BATCH_SIZE,
            per_device_eval_batch_size=SUBJECT_BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,
            num_train_epochs=MAX_EPOCHS,
            weight_decay=weight_decay,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            lr_scheduler_type="cosine",
            fp16=USE_FP16,
            seed=SEED,
            data_seed=SEED,
            remove_unused_columns=False,
        )

        model = CustomBERTModel(
            BASE_MODEL,
            class_weights=class_weights_np,
            chunk_micro_batch_size=CHUNK_MICRO_BATCH_SIZE,
            freeze_bottom_layers=FREEZE_BOTTOM_LAYERS,
            dropout_rate=dropout_rate,
            use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=fold_ds["train"],
            eval_dataset=fold_ds["validation"],
            data_collator=SubjectChunkingCollator(tokenizer, max_len=MAX_LEN, stride=STRIDE),
            compute_metrics=compute_metrics,
        )
        trainer.model_accepts_loss_kwargs = False

        trainer.train(resume_from_checkpoint=False)

        val_out = trainer.predict(fold_ds["validation"])
        val_pred, val_prob = extract_pred_and_prob(val_out.predictions)
        val_true = np.asarray(fold_ds["validation"]["target_class"], dtype=np.int64)

        train_out = trainer.predict(fold_ds["train"])
        train_pred, _ = extract_pred_and_prob(train_out.predictions)
        train_true = np.asarray(fold_ds["train"]["target_class"], dtype=np.int64)

        train_macro = float(f1_score(train_true, train_pred, labels=LABEL_SPACE, average="macro", zero_division=0))
        fold_macro = float(f1_score(val_true, val_pred, labels=LABEL_SPACE, average="macro", zero_division=0))

        fold_scores.append(fold_macro)
        fold_best_epochs.append(best_epoch_from_log(trainer.state.log_history))


        best_fold_score = max(best_fold_score, fold_macro)

        oof_true[va_idx] = val_true
        oof_pred[va_idx] = val_pred
        oof_prob[va_idx] = val_prob[:, 1]

        print(f"Fold {fold_idx} train macro F1: {train_macro:.4f}")
        print(f"Fold {fold_idx} validation macro F1: {fold_macro:.4f} (best epoch {fold_best_epochs[-1]})")
        print(f"Fold {fold_idx} train labels: {np.bincount(train_true, minlength=NUM_CLASSES).tolist()}")
        print(f"Fold {fold_idx} validation labels: {np.bincount(val_true, minlength=NUM_CLASSES).tolist()}")
        print(f"Fold {fold_idx} validation predictions: {np.bincount(val_pred, minlength=NUM_CLASSES).tolist()}")
        print(f"FOLD_RESULT fold={fold_idx} validation_size={len(va_idx)} macro_f1={fold_macro:.4f}")

        fold_rows.append(
            {
                "fold": fold_idx,
                "train_macro_f1": train_macro,
                "validation_macro_f1": fold_macro,
                "best_epoch": fold_best_epochs[-1],
            }
        )

        del trainer, model, val_out, train_out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        shutil.rmtree(fold_output_dir, ignore_errors=True)

    if np.any(oof_true < 0) or np.any(oof_pred < 0):
        raise RuntimeError("Incomplete OOF arrays detected.")
    if np.any(np.isnan(oof_prob)):
        raise RuntimeError("OOF probabilities contain NaN values.")

    oof_rows = []
    train_ids = train_dataset["train"]["id"]
    train_src = train_dataset["train"]["source"]
    train_score = train_dataset["train"]["continuous_score"]
    for i in range(len(train_ids)):
        oof_rows.append(
            {
                "id": str(train_ids[i]),
                "participant": participant_key(train_ids[i]),
                "source": str(train_src[i]),
                "continuous_score": safe_float(train_score[i]),
                "true_label": int(oof_true[i]),
                "pred_label": int(oof_pred[i]),
                "p_mild_moderate": float(1.0 - oof_prob[i]),
                "p_severe": float(oof_prob[i]),
            }
        )

    return {
        "fold_scores": fold_scores,
        "fold_best_epochs": fold_best_epochs,
        "oof_rows": oof_rows,
    }


def refit_on_full_data(efn_power: float, weight_decay: float, dropout_rate: float, config_dir: str, num_epochs: int) -> None:
    """Retrain once on 100% of the training data; this is the artifact used for inference."""
    reseed()
    full_ds = train_dataset["train"].map(lambda x: {**x, "target_class": int(x["labels"])})
    full_cls = np.asarray(full_ds["target_class"], dtype=np.int64)
    class_weights_np, _ = build_class_weights(full_cls, power=efn_power)

    refit_output_dir = os.path.join(config_dir, "refit_full")
    shutil.rmtree(refit_output_dir, ignore_errors=True)

    args = TrainingArguments(
        output_dir=refit_output_dir,
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="epoch",
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.08,
        per_device_train_batch_size=SUBJECT_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        lr_scheduler_type="cosine",
        fp16=USE_FP16,
        seed=SEED,
        data_seed=SEED,
        remove_unused_columns=False,
    )

    model = CustomBERTModel(
        BASE_MODEL,
        class_weights=class_weights_np,
        chunk_micro_batch_size=CHUNK_MICRO_BATCH_SIZE,
        freeze_bottom_layers=FREEZE_BOTTOM_LAYERS,
        dropout_rate=dropout_rate,
        use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=full_ds,
        data_collator=SubjectChunkingCollator(tokenizer, max_len=MAX_LEN, stride=STRIDE),
    )
    trainer.model_accepts_loss_kwargs = False
    trainer.train(resume_from_checkpoint=False)

    shutil.rmtree(BEST_MODEL_DIR, ignore_errors=True)
    trainer.save_model(BEST_MODEL_DIR)
    tokenizer.save_pretrained(BEST_MODEL_DIR)
    # 04_predict_epi.py loads model.safetensors / pytorch_model.bin directly
    try:
        from safetensors.torch import save_file
        save_file(trainer.model.state_dict(), os.path.join(BEST_MODEL_DIR, "model.safetensors"))
    except Exception as e:
        print(f"Warning: Could not save safetensors: {e}")

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    shutil.rmtree(refit_output_dir, ignore_errors=True)


# EPI evaluation moved to 04_predict_epi.py for separation of concerns


# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------


# Grid search removed - using hardcoded config (EFN_WEIGHT_POWER, WEIGHT_DECAY, DROPOUT_RATE)


# Train with hardcoded config on DAIC+PDCH (external training)
shutil.rmtree(BEST_MODEL_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

final_config_dir = os.path.join(OUTPUT_DIR, "final_config")
shutil.rmtree(final_config_dir, ignore_errors=True)
os.makedirs(final_config_dir, exist_ok=True)

final_cv_result = run_cv_for_config(
    EFN_WEIGHT_POWER,
    WEIGHT_DECAY,
    DROPOUT_RATE,
    final_config_dir,
    FINAL_N_FOLDS,
)
if not final_cv_result["fold_scores"]:
    raise RuntimeError("Cross-validation produced no folds.")

# CV is only the performance estimate; the shipped model is refit on all data.
_epochs = [e for e in final_cv_result["fold_best_epochs"] if e]
refit_epochs = int(np.median(_epochs)) if _epochs else MAX_EPOCHS
print(f"\nRefitting on full training data for {refit_epochs} epoch(s)...")
refit_on_full_data(EFN_WEIGHT_POWER, WEIGHT_DECAY, DROPOUT_RATE, final_config_dir, refit_epochs)
shutil.rmtree(final_config_dir, ignore_errors=True)
print(f"Saved full-data refit model: {BEST_MODEL_DIR}")

final_cv_mean = float(np.mean(final_cv_result["fold_scores"]))
final_cv_std = float(np.std(final_cv_result["fold_scores"]))
final_cv_best = float(np.max(final_cv_result["fold_scores"]))

# Save OOF predictions
write_rows_csv(
    SUBJECT_PREDICTIONS_CSV,
    [
        "id", "participant", "source", "continuous_score",
        "true_label", "pred_label", "p_mild_moderate", "p_severe",
    ],
    final_cv_result["oof_rows"],
)

print(f"Saved subject predictions: {SUBJECT_PREDICTIONS_CSV}")


# Extract OOF true and pred from oof_rows for metrics/plots
oof_true = np.array([int(r["true_label"]) for r in final_cv_result["oof_rows"]], dtype=np.int64)
oof_pred = np.array([int(r["pred_label"]) for r in final_cv_result["oof_rows"]], dtype=np.int64)
oof_prob = np.array([float(r["p_severe"]) for r in final_cv_result["oof_rows"]], dtype=np.float64)

# Compute severe-positive ROC-AUC and average precision on pooled OOF probabilities
try:
    roc_auc_severe = float(roc_auc_score(oof_true == 1, oof_prob))
    average_precision_severe = float(average_precision_score(oof_true == 1, oof_prob))
except Exception as e:
    print(f"Warning: Could not compute ROC-AUC: {e}")
    roc_auc_severe = np.nan
    average_precision_severe = np.nan

# Compute per-class metrics
val_precision = precision_score(oof_true, oof_pred, labels=LABEL_SPACE, average=None, zero_division=0)
val_recall = recall_score(oof_true, oof_pred, labels=LABEL_SPACE, average=None, zero_division=0)

write_single_row_csv(
    SEVERE_METRICS_CSV,
    {
        "positive_class": "severe",
        "support": int(np.sum(oof_true == 1)),
        "precision": float(val_precision[1]),
        "recall": float(val_recall[1]),
        "roc_auc": roc_auc_severe,
        "average_precision": average_precision_severe,
    },
)
print(f"Saved severe discrimination metrics: {SEVERE_METRICS_CSV}")

# Compute OOF-level accuracy and balanced accuracy
oof_accuracy = float(np.mean(oof_pred == oof_true))
oof_bal_acc = balanced_accuracy_score(oof_true, oof_pred)
oof_macro_f1 = float(f1_score(oof_true, oof_pred, labels=LABEL_SPACE, average="macro", zero_division=0))

# Compute confusion matrix
val_cm = confusion_matrix(oof_true, oof_pred, labels=LABEL_SPACE)

# Find best fold index
best_fold_idx = int(np.argmax(final_cv_result["fold_scores"]))
best_fold_number = best_fold_idx + 1

n_val_class0 = int(np.sum(oof_true == 0))
n_val_class1 = int(np.sum(oof_true == 1))

# Permutation test (subject-wise) - extract subject IDs from oof_rows
oof_subject_ids = np.asarray([str(row["id"]) for row in final_cv_result["oof_rows"]], dtype=object)
perm_result = permutation_participant_wise(oof_true, oof_pred, participant_ids=oof_subject_ids, n_permutations=10000, seed=SEED, label_space=LABEL_SPACE)

# Update training summary to match 02 format
final_summary = {
    "n": int(len(oof_true)),
    "cv_estimator": "full_cv_oof",
    "best_fold": best_fold_number,
    "n_folds": FINAL_N_FOLDS,
    "accuracy": oof_accuracy,
    "balanced_accuracy": oof_bal_acc,
    "macro_f1": oof_macro_f1,
    "roc_auc_severe": roc_auc_severe,
    "average_precision_severe": average_precision_severe,
    "support_mild_moderate": n_val_class0,
    "support_severe": n_val_class1,
    **{f"fold_{i+1}_macro_f1": float(final_cv_result["fold_scores"][i]) for i in range(FINAL_N_FOLDS)},
    "best_fold_macro_f1": final_cv_best,
    "cv_macro_f1_mean": final_cv_mean,
    "cv_macro_f1_std": final_cv_std,
    "learning_rate": 3e-5,
    "weight_decay": float(WEIGHT_DECAY),
    "freeze_bottom_layers": 4,
    "dropout_rate": float(DROPOUT_RATE),
    "class_weight_power": float(EFN_WEIGHT_POWER),
}
for i in range(len(CLASS_NAMES)):
    final_summary[f"precision_{CLASS_NAMES[i]}"] = float(val_precision[i])
    final_summary[f"recall_{CLASS_NAMES[i]}"] = float(val_recall[i])
final_summary["confusion_matrix"] = str(val_cm.tolist())
final_summary["perm_observed_macro_f1"] = perm_result["observed_macro_f1"]
final_summary["perm_null_macro_f1_mean"] = perm_result["null_macro_f1_mean"]
final_summary["perm_p_value"] = perm_result["perm_p_value"]
final_summary["perm_n_permutations"] = perm_result["n_permutations"]

write_single_row_csv(VALIDATION_METRICS_CSV, final_summary)

# Save confusion matrix plot
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(val_cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(CLASS_NAMES)
ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"DAIC: {FINAL_N_FOLDS}-Fold CV (pooled out-of-fold)")
for i in range(len(LABEL_SPACE)):
    for j in range(len(LABEL_SPACE)):
        ax.text(j, i, str(val_cm[i, j]), ha="center", va="center", fontsize=14, fontweight="bold")
plt.colorbar(im, ax=ax)
plt.tight_layout()
os.makedirs(PLOT_DIR, exist_ok=True)
confusion_matrix_path = os.path.join(PLOT_DIR, "confusion_matrix.png")
plt.savefig(confusion_matrix_path, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved confusion matrix plot: {confusion_matrix_path}")

roc_path = os.path.join(PLOT_DIR, "roc_precision_recall_severe.png")
save_roc_and_pr_curves(oof_true, oof_prob, roc_path)
print(f"Saved pooled OOF severe ROC and precision-recall curves: {roc_path}")



print("\n" + "=" * 70)
print(f"TRAINING COMPLETE: {FINAL_N_FOLDS}-FOLD CV ON DAIC")
print("=" * 70)
print(f"Hyperparameters:")
print(f"  efn_weight_power: {EFN_WEIGHT_POWER}")
print(f"  weight_decay: {WEIGHT_DECAY}")
print(f"  dropout_rate: {DROPOUT_RATE}")
print(f"\nResults ({FINAL_N_FOLDS}-fold CV):")
print(f"  macro_f1_mean: {final_cv_mean:.4f} (±{final_cv_std:.4f})")
print(f"  roc_auc_severe: {roc_auc_severe:.4f}")
print(f"  average_precision_severe: {average_precision_severe:.4f}")
print(f"  best_fold_macro_f1: {final_cv_best:.4f}")
print(f"  fold_scores: {final_cv_result['fold_scores']}")
print(f"\nOutputs:")
print(f"  Best model: {BEST_MODEL_DIR}")
print(f"  Subject predictions: {SUBJECT_PREDICTIONS_CSV}")
print(f"  Validation metrics: {VALIDATION_METRICS_CSV}")
print(f"\nNext step - evaluate on EPI:")
print(f"  python 04_predict_epi.py --model_dir {BEST_MODEL_DIR}")
print("=" * 70)
