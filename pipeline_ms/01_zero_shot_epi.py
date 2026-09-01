import os
import csv
import re
import random
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, roc_curve, auc, precision_score, recall_score, average_precision_score, precision_recall_curve

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Georgia", "DejaVu Serif", "Times New Roman", "serif"]

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Config and Setup
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Edit stage paths here. This stage evaluates a fresh base model without supervised training.
PROJECT_ROOT = Path("/zi/home/luke.bregulla/Desktop/DSS")
TEST_FILE = str(PROJECT_ROOT / "data_ms/data_epi.csv")
TEST_SEP = "auto"
TEST_ENCODING = "utf-8"
TEXT_INPUT = "full_transcript"
SOURCE_DROPOUT_RATE = 0.3
TRAIN_OUTPUT_DIR = None
ZERO_SHOT_OUTPUT_DIR = str(PROJECT_ROOT / "results_ms/01_zero_shot_epi")
OUTPUT_NAME = "epi_zero_shot"
SUBJECT_PREDICTIONS_CSV = os.path.join(ZERO_SHOT_OUTPUT_DIR, "subject_predictions.csv")
SEVERE_METRICS_CSV = os.path.join(ZERO_SHOT_OUTPUT_DIR, "severe_discrimination_metrics.csv")
NUM_CLASSES = 2
MODEL_NUM_CLASSES = NUM_CLASSES

BASE_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"

# DeBERTa v3-base trained explicitly for zero-shot classification via NLI (Laurer et al. 2023, arxiv:2312.17543)
MAX_LEN = 512
STRIDE = 256
CHUNK_MICRO_BATCH_SIZE = 8
SUBJECT_BATCH_SIZE = 2
SEED = 42

# Keep inference deterministic where the runtime uses CUDA kernels.
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Map evaluation classes to label indices
METRIC_LABELS = [0, 1]
if METRIC_LABELS is None:
    raise ValueError("Binary zero-shot evaluation requires label indices [0, 1].")


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Utility Functions
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def prepare_model_text(dataset):
    """All runs use the full transcript view; patient-only mode is intentionally disabled."""
    return dataset

# generic csv writer
def write_csv(output_path, fieldnames, rows):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if isinstance(rows, dict):
            writer.writerow(rows)
        else:
            writer.writerows(rows)


def save_roc_and_pr_curves(y_true, prob_severe, output_path):
    """Save severe-positive ROC and precision-recall curves."""
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    prob_severe = np.asarray(prob_severe, dtype=float)
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
    axes[1].axhline(severe_true.mean(), linestyle="--", color="gray", linewidth=1.2, label="Severe prevalence")
    axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision", title="Precision-Recall: Severe vs Mild/Moderate")
    axes[1].legend(loc="best", frameon=False)
    axes[1].grid(alpha=0.2)

    fig.suptitle("Zero-Shot Evaluation", y=1.03)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return roc_auc, average_precision

# permutation test for macro F1 vs chance
def macro_f1_present_labels(y_true, y_pred, label_space=None):
    """Compute macro F1 on the labels present in y_true or specified in label_space."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size == 0:
        return 0.0
    labels = np.asarray(np.unique(y_true), dtype=np.int64) if label_space is None else np.asarray(label_space, dtype=np.int64)
    return float(f1_score(y_true, y_pred, labels=labels.tolist(), average="macro", zero_division=0))

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
        "n_samples": int(y_true.size),
        "observed_macro_f1": observed,
        "null_macro_f1_mean": float(np.mean(perm_scores)),
        "p": float((np.sum(perm_scores >= observed) + 1) / (n_permutations + 1)),
        "n_permutations": int(n_permutations),
        "n_participants": int(len(blocks)),
        "permutation_unit": "participant",
        "label_space": "|".join([str(int(x)) for x in label_space.tolist()]),
    }


# plot class distribution
def save_class_distribution_plot(distribution_map, output_path):
    labels = ["mild_moderate", "severe"]
    dataset_items = list(distribution_map.items())

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    x = np.arange(len(labels), dtype=float)
    
    for i, (name, counts) in enumerate(dataset_items):
        counts = np.asarray(counts, dtype=int)
        bars = ax.bar(x + i * 0.35, counts, width=0.35, label=name, alpha=0.9)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.5, f"{int(h)}", ha="center", va="bottom", fontsize=8)
    
    ax.set_xticks(x + 0.175)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of cases")
    ax.set_title("Test Set Class Distribution")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Load Test Data
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print(f"Loading test data from: {TEST_FILE}")
if TEST_SEP in ("auto", "", None):
    with open(TEST_FILE, "r", encoding=TEST_ENCODING) as f:
        header = f.readline()
    TEST_SEP = "\t" if header.count("\t") > header.count(",") else ","
    print(f"Detected separator: {TEST_SEP!r}")
test_dataset = load_dataset("csv", data_files={"test": TEST_FILE}, sep=TEST_SEP, encoding=TEST_ENCODING)["test"]
print(f"Test set size: {len(test_dataset)} | columns: {test_dataset.column_names}")

# The prepared EPI file is the source of truth for holdout labels.
if "labels" not in test_dataset.column_names:
    raise ValueError(
        f"Required 'labels' column not found in {TEST_FILE}. "
        f"Columns read with separator {TEST_SEP!r}: {test_dataset.column_names}"
    )
test_true_classes = np.asarray(test_dataset["labels"], dtype=np.int64)

# Extract IDs before filtering
if "id" in test_dataset.column_names:
    test_row_ids = list(test_dataset["id"])
elif "session_id" in test_dataset.column_names:
    test_row_ids = list(test_dataset["session_id"])
else:
    test_row_ids = [str(i) for i in range(len(test_dataset))]

# Filter to evaluation labels if not using all 3 classes
if NUM_CLASSES < MODEL_NUM_CLASSES:
    eval_labels = np.array(METRIC_LABELS, dtype=np.int64)
    mask = np.isin(test_true_classes, eval_labels)
    test_dataset = test_dataset.select(np.where(mask)[0].tolist())
    test_true_classes = test_true_classes[mask]
    test_row_ids = [test_row_ids[i] for i in np.where(mask)[0]]
    print(f"After filtering to evaluation_labels {tuple(eval_labels)}: {len(test_dataset)} samples remain")

# Keep labels, IDs, and continuous scores alongside the text for final association output.
test_continuous_scores = np.asarray(test_dataset["continuous_score"], dtype=float)
test_dataset = prepare_model_text(test_dataset)
test_dataset = test_dataset.select_columns(["text"])
print(f"Final test set size: {len(test_dataset)}")


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Model Classes (from Train.py)
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class CustomBERTModel(nn.Module):

    def __init__(self, pretrained_model_name, class_weights=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.config = self.encoder.config
        self.config.num_labels = MODEL_NUM_CLASSES
        self.chunk_micro_batch_size = CHUNK_MICRO_BATCH_SIZE
        self.dropout = nn.Dropout(SOURCE_DROPOUT_RATE)
        self.classifier = nn.Linear(self.config.hidden_size, MODEL_NUM_CLASSES)
        if class_weights is None:
            class_weights = np.ones(MODEL_NUM_CLASSES, dtype=np.float32)
        self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))

    def sequential_chunking(self, input_ids, attention_mask):
        cls_parts = []
        total_chunks = input_ids.size(0)
        for start in range(0, total_chunks, self.chunk_micro_batch_size):
            end = min(start + self.chunk_micro_batch_size, total_chunks)
            out = self.encoder(
                input_ids=input_ids[start:end],
                attention_mask=attention_mask[start:end],
            )
            cls_embeddings = self.dropout(out.last_hidden_state[:, 0, :])
            cls_parts.append(cls_embeddings)
        return torch.cat(cls_parts, dim=0)

    def forward(self, input_ids=None, attention_mask=None, labels=None, subject_chunk_counts=None):
        loss = None

        if subject_chunk_counts is not None:
            subject_logits_list = []
            subject_labels_list = []
            chunk_idx = 0

            for subject_idx, num_chunks in enumerate(subject_chunk_counts):
                subj_input_ids = input_ids[chunk_idx: chunk_idx + num_chunks]
                subj_attention = attention_mask[chunk_idx: chunk_idx + num_chunks]
                cls_embeddings = self.sequential_chunking(subj_input_ids, subj_attention)

                pooled_cls = torch.mean(cls_embeddings, dim=0)
                subject_logit = self.classifier(pooled_cls)
                subject_logits_list.append(subject_logit)

                if labels is not None:
                    subject_labels_list.append(labels[subject_idx])

                chunk_idx += num_chunks

            subject_logits = torch.stack(subject_logits_list)

            if labels is not None:
                return {
                    "loss": torch.nn.functional.cross_entropy(subject_logits, labels.long(), weight=self.class_weights),
                    "logits": subject_logits,
                }
            return {
                "logits": subject_logits,
            }

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = self.dropout(outputs.last_hidden_state[:, 0, :])
        chunk_logits = self.classifier(cls_embedding)
        if labels is not None:
            return {
                "loss": torch.nn.functional.cross_entropy(chunk_logits, labels.long(), weight=self.class_weights),
                "logits": chunk_logits,
            }
        return {
            "logits": chunk_logits,
        }

# chunking and tracking for subject level aggregation
class SubjectChunkingCollator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_len = MAX_LEN
        self.stride = STRIDE

    def __call__(self, batch):
        all_input_ids = []
        all_attention_masks = []
        subject_labels = []
        subject_chunk_counts = []
        has_targets = "target_class" in batch[0]

        for example in batch:
            text = str(example["text"]).lower()
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
                subject_labels.append(int(example["target_class"]))
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

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Binary NLI zero-shot inference
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print(f"\nLoading DeBERTa zero-shot model: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL)

# This model is trained for entailment vs not-entailment (2-class zero-shot NLI)
if model.config.num_labels not in [2, 3]:
    raise RuntimeError(f"Expected 2-3 labels for zero-shot, got {model.config.num_labels}: {model.config.id2label}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Score exactly the two study hypotheses.
candidate_labels = ["mild or moderate depression", "severe depression"]
hypothesis_template = "This patient has {}."

# Find entailment label (typically 2 for 3-label models, or 1 for 2-label)
entailment_id = None
for label_id, label_name in model.config.id2label.items():
    if "entailment" in label_name.lower():
        entailment_id = int(label_id)
        break

if entailment_id is None:
    raise ValueError(f"Could not find entailment label in {model.config.id2label}")


def zero_shot_transcript_scores(text):
    chunks = tokenizer(
        str(text).lower(),
        truncation=True,
        max_length=MAX_LEN,
        stride=STRIDE,
        return_overflowing_tokens=True,
    )
    premises = tokenizer.batch_decode(chunks["input_ids"], skip_special_tokens=True)
    scores = []
    with torch.no_grad():
        for candidate in candidate_labels:
            hypothesis = hypothesis_template.format(candidate)
            pairs = tokenizer(
                premises,
                [hypothesis] * len(premises),
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            pairs = {key: value.to(device) for key, value in pairs.items()}
            logits = model(**pairs).logits
            # Extract entailment score (higher = stronger support for this hypothesis)
            support_scores = torch.softmax(logits, dim=-1)[:, entailment_id]
            scores.append(float(support_scores.mean().cpu()))
    scores = np.asarray(scores, dtype=float)
    scores = scores / max(float(scores.sum()), 1e-12)
    return scores, scores, int(np.argmax(scores))


print(f"\nRunning binary DeBERTa zero-shot inference on {len(test_dataset)} samples...")
zero_shot_results = [zero_shot_transcript_scores(text) for text in test_dataset["text"]]
test_pred_probs = np.vstack([result[0] for result in zero_shot_results])
test_pred_margins = np.vstack([result[1] for result in zero_shot_results])
test_pred_classes = np.asarray([result[2] for result in zero_shot_results], dtype=np.int64)
print("Predictions complete")


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Metrics and Output Files
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Compute confusion matrix and class metrics
test_cm = confusion_matrix(test_true_classes, test_pred_classes, labels=METRIC_LABELS).astype(float)
recalls = [(test_cm[i, i] / test_cm[i, :].sum()) if test_cm[i, :].sum() > 0 else 0.0 for i in range(len(METRIC_LABELS))]
test_bal_acc = np.mean(recalls) if recalls else 0.0
test_macro_f1 = f1_score(test_true_classes, test_pred_classes, labels=METRIC_LABELS, average="macro", zero_division=0)
test_precision = precision_score(test_true_classes, test_pred_classes, labels=METRIC_LABELS, average=None, zero_division=0)
test_recall = recall_score(test_true_classes, test_pred_classes, labels=METRIC_LABELS, average=None, zero_division=0)
try:
    roc_auc_severe = float(roc_auc_score(test_true_classes == 1, test_pred_probs[:, 1]))
    average_precision_severe = float(average_precision_score(test_true_classes == 1, test_pred_probs[:, 1]))
except:
    roc_auc_severe = float("nan")
    average_precision_severe = float("nan")

print("\n" + "="*60)
print("TEST RESULTS")
print("="*60)
print(f"Balanced Accuracy: {test_bal_acc:.4f}")
print(f"Macro F1:          {test_macro_f1:.4f}")
print(f"ROC-AUC severe:    {roc_auc_severe:.4f}")
print(f"Average precision severe: {average_precision_severe:.4f}")
print("Confusion Matrix (rows=true, cols=pred):")
print(test_cm)

# Permute whole participant blocks, not individual EPI sessions.
perm_result = permutation_participant_wise(test_true_classes, test_pred_classes, participant_ids=np.asarray(test_row_ids), n_permutations=10000, seed=42, label_space=METRIC_LABELS)
test_p_value = perm_result["p"]
test_null_mean = perm_result["null_macro_f1_mean"]
print("Participant-level permutation test:")
print(f"  n_permutations: {perm_result['n_permutations']}, null macro F1 mean: {test_null_mean:.4f}, p-value: {test_p_value:.4f}")

# Save all output files
os.makedirs(ZERO_SHOT_OUTPUT_DIR, exist_ok=True)

# individual subject predictions
subject_pred_path = SUBJECT_PREDICTIONS_CSV
rows = [
    {
        "id": str(row_id),
        "participant": participant_key(row_id),
        "true_label": int(y_t),
        "pred_label": int(y_p),
        "p_mild_moderate": float(test_pred_probs[i, 0]),
        "p_severe": float(test_pred_probs[i, 1]),
        "binary_score_mild_moderate": float(test_pred_margins[i, 0]),
        "binary_score_severe": float(test_pred_margins[i, 1]),
        "continuous_score": float(test_continuous_scores[i]),
    }
    for i, (row_id, y_t, y_p) in enumerate(zip(test_row_ids, test_true_classes, test_pred_classes))
]
write_csv(
    subject_pred_path,
    [
        "id", "participant", "true_label", "pred_label", "p_mild_moderate", "p_severe",
        "binary_score_mild_moderate", "binary_score_severe", "continuous_score",
    ],
    rows,
)
print(f"✓ Saved: {subject_pred_path}")

write_csv(
    SEVERE_METRICS_CSV,
    ["positive_class", "support", "precision", "recall", "roc_auc", "average_precision"],
    {
        "positive_class": "severe",
        "support": int(np.sum(test_true_classes == 1)),
        "precision": float(test_precision[1]),
        "recall": float(test_recall[1]),
        "roc_auc": roc_auc_severe,
        "average_precision": average_precision_severe,
    },
)
print(f"✓ Saved: {SEVERE_METRICS_CSV}")

# overall metrics
overall_metrics_path = os.path.join(ZERO_SHOT_OUTPUT_DIR, f"{OUTPUT_NAME}_overall_metrics.csv")
write_csv(
    overall_metrics_path,
    ["n", "balanced_accuracy", "macro_f1", "roc_auc_severe", "average_precision_severe", "perm_observed_macro_f1", "perm_null_macro_f1_mean", "perm_p_value", "perm_n_permutations"],
    {
        "n": len(test_true_classes),
        "balanced_accuracy": test_bal_acc,
        "macro_f1": test_macro_f1,
        "roc_auc_severe": roc_auc_severe,
        "average_precision_severe": average_precision_severe,
        "perm_observed_macro_f1": perm_result["observed_macro_f1"],
        "perm_null_macro_f1_mean": test_null_mean,
        "perm_p_value": test_p_value,
        "perm_n_permutations": perm_result["n_permutations"],
    },
)
print(f"✓ Saved: {overall_metrics_path}")

# CM
perf_dir = os.path.join(ZERO_SHOT_OUTPUT_DIR, "performance_plots")
os.makedirs(perf_dir, exist_ok=True)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(test_cm, cmap="Blues")
class_labels = ["mild_moderate", "severe"]
ax.set_xticks(range(len(METRIC_LABELS)))
ax.set_yticks(range(len(METRIC_LABELS)))
ax.set_xticklabels(class_labels)
ax.set_yticklabels(class_labels)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Test (out-of-domain)")
for i in range(len(METRIC_LABELS)):
    for j in range(len(METRIC_LABELS)):
        ax.text(j, i, str(int(test_cm[i, j])), ha="center", va="center", color="white" if test_cm[i, j] > test_cm.max()/2 else "black")
plt.colorbar(im, ax=ax)
fig.tight_layout()
cm_path = os.path.join(perf_dir, "confusion_matrix.png")
fig.savefig(cm_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"✓ Saved: {cm_path}")

# Plot severe-positive ROC and precision-recall curves
roc_path = os.path.join(perf_dir, "roc_precision_recall_severe.png")
save_roc_and_pr_curves(test_true_classes, test_pred_probs[:, 1], roc_path)
print(f"✓ Saved: {roc_path}")

# plot Class distribution
class_dist = {"test": np.bincount(np.asarray(test_true_classes, dtype=np.int64), minlength=NUM_CLASSES)}
class_dist_path = os.path.join(perf_dir, "class_distribution.png")
save_class_distribution_plot(class_dist, class_dist_path)
print(f"✓ Saved: {class_dist_path}")

print("Zero-shot EPI evaluation complete; supervised adaptation is disabled for this stage.")
raise SystemExit(0)


print("\n" + "="*60)
print("INFERENCE COMPLETE")
print("="*60)
