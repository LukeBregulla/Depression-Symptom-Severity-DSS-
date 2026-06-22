import os
import gc
import csv
import re
import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, DataCollatorForLanguageModeling, AutoModelForMaskedLM
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, roc_curve, auc, precision_score, recall_score
from safetensors.torch import load_file as load_safetensors

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Georgia", "DejaVu Serif", "Times New Roman", "serif"]

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Paths
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Data Loading
TRAIN_FILE = "/zi/home/luke.bregulla/Desktop/DSS/data/data_daic_pdch_3class_texts.csv"

OUTPUT_DIR = "/zi/home/luke.bregulla/Desktop/DSS/results/Validation"
LOGGING_DIR = "/zi/home/luke.bregulla/Desktop/DSS/results/Validation/logs"
PERFORMANCE_DIR = "/zi/home/luke.bregulla/Desktop/DSS/results/Validation/performance_plots"
VAL_PER_SUBJECT_PREDICTIONS_CSV_PATH = os.path.join(OUTPUT_DIR, "validation_subject_predictions_V12.csv")
BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model")
DATA_DISTRIBUTION_PLOT_PATH = os.path.join(PERFORMANCE_DIR, "training_class_distribution_train.png")


os.environ["TENSORBOARD_LOGGING_DIR"] = LOGGING_DIR

# model specifics
BASE_MODEL = "FacebookAI/xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
DOMAIN_ADAPTED_MODEL_DIR = os.path.join(OUTPUT_DIR, "domain_adapted_bert")
SAVE_PLOTS = True


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Configs
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=3e-5,
    warmup_ratio=0.08,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    num_train_epochs=8,
    weight_decay=0.09,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    lr_scheduler_type="cosine",
    fp16=torch.cuda.is_available(),
    seed=SEED,
    data_seed=SEED,
    remove_unused_columns=False,
)

MAX_LEN = 512
STRIDE = 256  
USE_GRADIENT_CHECKPOINTING = False
FREEZE_BOTTOM_LAYERS = 4
DROPOUT_RATE = 0.3
NUM_CLASSES = 3
CLASS_BALANCE_BETA = 0.999
EFN_WEIGHT_POWER = 1.0


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Functions and Helpers
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def build_class_weights(
    target_classes,
    num_classes=NUM_CLASSES,
    beta=CLASS_BALANCE_BETA,
    power=EFN_WEIGHT_POWER,
):
    """Class-balanced weights using effective number of samples."""
    cls = np.asarray(target_classes, dtype=np.int64).reshape(-1)
    counts = np.bincount(cls, minlength=num_classes).astype(np.float64)
    effective_num = np.zeros_like(counts)
    nonzero = counts > 0
    effective_num[nonzero] = (1.0 - np.power(beta, counts[nonzero])) / (1.0 - beta)

    weights = np.zeros_like(counts)
    weights[nonzero] = 1.0 / np.maximum(effective_num[nonzero], 1e-8)
    if np.any(nonzero):
        weights[nonzero] = weights[nonzero] / np.mean(weights[nonzero])
        weights[nonzero] = np.power(weights[nonzero], power)
        weights[nonzero] = weights[nonzero] / np.mean(weights[nonzero])
    return weights.astype(np.float32), counts.astype(np.int64)


def balanced_accuracy(y_true, y_pred):
    """Balanced accuracy on a fixed 3-class label space [0,1,2]."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).astype(float)
    recalls = []
    for i in range(3):
        denom = cm[i, :].sum()
        recalls.append((cm[i, i] / denom) if denom > 0 else 0.0)
    return float(np.mean(recalls))


def write_overall_metrics_csv(output_path, metrics_row):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
            ],
        )
        writer.writeheader()
        writer.writerow(metrics_row)


def write_per_session_metrics_csv(output_path, session_labels, true_classes, pred_classes):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    unique_sessions = sorted(set(session_labels), key=lambda s: (s == "unknown", s))
    rows = []
    true_classes = np.asarray(true_classes, dtype=np.int64)
    pred_classes = np.asarray(pred_classes, dtype=np.int64)

    for session_label in unique_sessions:
        mask = np.asarray([label == session_label for label in session_labels], dtype=bool)
        y_true = true_classes[mask]
        y_pred = pred_classes[mask]
        if y_true.size == 0:
            continue
        rows.append({
            "session": session_label,
            "n_samples": int(y_true.size),
            "accuracy": float(np.mean(y_pred == y_true)),
            "balanced_accuracy": balanced_accuracy(y_true, y_pred),
            "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["session", "n_samples", "accuracy", "balanced_accuracy", "macro_f1"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_auc_details_csv(output_path, rows):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_id",
                "class_count",
                "auc_ovr",
                "macro_auc_ovr",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_single_row_csv(output_path, row):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_subject_predictions_csv(output_path, row_ids, true_classes, pred_classes, pred_probs):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pred_probs = np.asarray(pred_probs, dtype=float)
    if len(row_ids) != len(true_classes):
        raise ValueError("row_ids length must match true_classes length")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "y_true",
                "y_pred",
                "p_none",
                "p_moderate",
                "p_severe",
            ],
        )
        writer.writeheader()
        for i, (row_id, y_t, y_p) in enumerate(zip(row_ids, true_classes, pred_classes)):
            writer.writerow(
                {
                    "index": str(row_id),
                    "y_true": int(y_t),
                    "y_pred": int(y_p),
                    "p_none": float(pred_probs[i, 0]),
                    "p_moderate": float(pred_probs[i, 1]),
                    "p_severe": float(pred_probs[i, 2]),
                }
            )


def macro_f1_present_labels(y_true, y_pred, label_space=None):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size == 0:
        return 0.0
    labels = np.asarray(np.unique(y_true), dtype=np.int64) if label_space is None else np.asarray(label_space, dtype=np.int64)
    return float(f1_score(y_true, y_pred, labels=labels.tolist(), average="macro", zero_division=0))


def macro_f1_permutation_vs_chance(y_true, y_pred, subject_ids=None, n_permutations=10000, seed=SEED):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if subject_ids is None:
        subject_ids = np.arange(y_true.size, dtype=np.int64)
    subject_ids = np.asarray(subject_ids)
    if subject_ids.size != y_true.size:
        raise ValueError("subject_ids length must match y_true length")

    label_space = np.unique(y_true)
    observed = macro_f1_present_labels(y_true, y_pred, label_space=label_space)

    unique_subjects, first_idx = np.unique(subject_ids, return_index=True)
    subject_true = y_true[first_idx]
    index_by_subject = {sid: np.where(subject_ids == sid)[0] for sid in unique_subjects}

    rng = np.random.default_rng(seed)
    perm_scores = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        shuffled_subject_true = rng.permutation(subject_true)
        y_perm = np.empty_like(y_true)
        for sid, shuffled_label in zip(unique_subjects, shuffled_subject_true):
            y_perm[index_by_subject[sid]] = shuffled_label
        perm_scores[i] = macro_f1_present_labels(y_perm, y_pred, label_space=label_space)

    null_min = float(np.min(perm_scores))
    null_max = float(np.max(perm_scores))

    p_value_right = float((np.sum(perm_scores >= observed) + 1) / (n_permutations + 1))
    return {
        "n_samples": int(y_true.size),
        "observed_macro_f1": observed,
        "null_macro_f1_mean": float(np.mean(perm_scores)),
        "null_macro_f1_std": float(np.std(perm_scores)),
        "null_macro_f1_min": null_min,
        "null_macro_f1_max": null_max,
        "null_macro_f1_range": f"{null_min:.6f} to {null_max:.6f}",
        "p_value_right_tailed": p_value_right,
        "n_permutations": int(n_permutations),
        "n_subjects": int(unique_subjects.size),
        "permutation_unit": "subject",
        "label_space": "|".join([str(int(x)) for x in label_space.tolist()]),
        "note": "Permutation test (subject-wise): subject labels are permuted against fixed predictions and broadcast to all sessions of each subject; right-tailed p for better-than-chance macro-F1 on the labels present in the holdout set.",
    }


def save_ovr_roc_panel(y_true, y_scores, macro_auc, output_path):
    """Save a single 3-panel one-vs-rest ROC figure for all classes."""
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_scores = np.asarray(y_scores, dtype=float)
    if y_scores.ndim != 2 or y_scores.shape[1] != 3:
        raise ValueError("y_scores must have shape (n_samples, 3)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, sharey=True)
    class_aucs = {}

    for class_id, ax in enumerate(axes):
        y_true_bin = (y_true == class_id).astype(np.int64)
        y_score = y_scores[:, class_id]

        has_pos = int(np.sum(y_true_bin == 1)) > 0
        has_neg = int(np.sum(y_true_bin == 0)) > 0
        if not (has_pos and has_neg):
            ax.text(0.5, 0.5, f"Class {class_id}\ninsufficient labels", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Class {class_id} vs Rest")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2)
            ax.grid(alpha=0.2)
            class_aucs[class_id] = float("nan")
            continue

        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        class_auc = float(auc(fpr, tpr))
        class_aucs[class_id] = class_auc

        ax.plot(fpr, tpr, color="#1F77B4", linewidth=2.0, label=f"AUC = {class_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Class {class_id} vs Rest")
        ax.legend(loc="lower right", frameon=False)
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("True Positive Rate")
    for ax in axes:
        ax.set_xlabel("False Positive Rate")

    fig.suptitle(f"One-vs-Rest ROC Curves (Macro AUC = {macro_auc:.3f})", y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return class_aucs


def hamd_to_three_classes(hamd_values):
    hamd_values = np.asarray(hamd_values, dtype=float)
    return np.where(hamd_values <= 7, 0, np.where(hamd_values <= 23, 1, 2)).astype(np.int64)


def extract_role_text(full_text, role):
    text = "" if full_text is None else str(full_text)
    pattern = rf"{role}\s*:\s*(.*?)(?=(?:therapist|patient)\s*:|$)"
    matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not matches:
        return ""
    return " ".join(m.strip() for m in matches if m and m.strip())


def count_words(text):
    return len(re.findall(r"\b\w+\b", text))


def count_tokens(text):
    stripped = text.strip()
    if not stripped:
        return 0
    return len(tokenizer(stripped, add_special_tokens=False)["input_ids"])


def build_role_stats_rows(dataset_name, texts):
    rows = []
    for role in ["therapist", "patient"]:
        per_case_words = []
        per_case_tokens = []
        for text in texts:
            role_text = extract_role_text(text, role)
            per_case_words.append(count_words(role_text))
            per_case_tokens.append(count_tokens(role_text))

        words_arr = np.asarray(per_case_words, dtype=float)
        tokens_arr = np.asarray(per_case_tokens, dtype=float)
        rows.append(
            {
                "dataset": dataset_name,
                "role": role,
                "n_cases": int(len(texts)),
                "avg_words_per_case": float(np.mean(words_arr)) if words_arr.size else 0.0,
                "std_words_per_case": float(np.std(words_arr)) if words_arr.size else 0.0,
                "avg_tokens_per_case": float(np.mean(tokens_arr)) if tokens_arr.size else 0.0,
                "std_tokens_per_case": float(np.std(tokens_arr)) if tokens_arr.size else 0.0,
            }
        )
    return rows


def save_role_stats_csv(output_path, rows):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "role",
                "n_cases",
                "avg_words_per_case",
                "std_words_per_case",
                "avg_tokens_per_case",
                "std_tokens_per_case",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_class_distribution_plot(distribution_map, output_path):
    labels = ["none", "moderate", "severe"]
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
        fig.suptitle("Class Distribution: Train (left) vs Test (right)")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def extract_exact_ids(split_dataset, preferred_columns=("id", "session_id", "csv_id")):
    for col in preferred_columns:
        if col in split_dataset.column_names:
            return [str(x) for x in split_dataset[col]]
    return [str(i) for i in range(len(split_dataset))]


def parse_session_labels(session_ids):
    session_labels = []
    for raw_id in session_ids:
        raw_id = str(raw_id)
        if "-" in raw_id:
            _, session_label = raw_id.rsplit("-", 1)
            session_labels.append(session_label)
        else:
            session_labels.append("unknown")
    return session_labels


def parse_subject_ids(session_ids):
    subject_ids = []
    for raw_id in session_ids:
        raw_id = str(raw_id)
        if "-" in raw_id:
            subject_id, _ = raw_id.rsplit("-", 1)
            subject_ids.append(subject_id)
        else:
            subject_ids.append(raw_id)
    return subject_ids


def session_sort_key(session_label):
    if session_label is None:
        return (1, float("inf"), "")
    s = str(session_label)
    match = re.search(r"(\d+)", s)
    if match:
        return (0, int(match.group(1)), s)
    return (1, float("inf"), s)


def compute_lag1_autocorrelation(subject_ids, session_labels, values):
    """Lag-1 autocorrelation built from consecutive within-subject sessions."""
    subject_ids = np.asarray(subject_ids)
    session_labels = np.asarray(session_labels)
    values = np.asarray(values, dtype=float)

    if subject_ids.size == 0 or values.size == 0 or subject_ids.size != values.size:
        return float("nan"), 0

    x_prev = []
    x_curr = []
    for sid in np.unique(subject_ids):
        idx = np.where(subject_ids == sid)[0]
        if idx.size < 2:
            continue
        ordered_idx = sorted(idx.tolist(), key=lambda i: session_sort_key(session_labels[i]))
        series = values[ordered_idx]
        x_prev.extend(series[:-1].tolist())
        x_curr.extend(series[1:].tolist())

    if len(x_prev) < 2:
        return float("nan"), len(x_prev)

    x_prev = np.asarray(x_prev, dtype=float)
    x_curr = np.asarray(x_curr, dtype=float)
    if np.std(x_prev) <= 1e-12 or np.std(x_curr) <= 1e-12:
        return float("nan"), int(len(x_prev))

    return float(np.corrcoef(x_prev, x_curr)[0, 1]), int(len(x_prev))


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Dataset loading and splitting
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

train_dataset = load_dataset("csv", data_files={"train": TRAIN_FILE})["train"]

# Paper artifacts: dataset class distribution and role-wise text statistics.
train_labels_pre_split = np.asarray(train_dataset["labels"], dtype=np.int64)
train_class_counts_full = np.bincount(train_labels_pre_split, minlength=3).astype(int)

class_distribution = {
    "Train (full, pre-split)": train_class_counts_full,
}
save_class_distribution_plot(class_distribution, DATA_DISTRIBUTION_PLOT_PATH)
print(f"Saved dataset class distribution plot: {DATA_DISTRIBUTION_PLOT_PATH}")
print(
    "Pre-split train class distribution (none / moderate / severe): "
    f"train={train_class_counts_full.tolist()}"
)


train_keep_cols = ["text", "labels"]
if "id" in train_dataset.column_names:
    train_keep_cols = ["id"] + train_keep_cols
train_dataset = train_dataset.select_columns(train_keep_cols)

# Split into train(80%) / validation(20%)
train_indices, val_indices = train_test_split(
    np.arange(len(train_dataset)),
    test_size=0.2,
    stratify=np.asarray(train_dataset["labels"], dtype=np.int64),
    random_state=SEED,
)

dataset = DatasetDict({
    "train": train_dataset.select(train_indices.tolist()),
    "validation": train_dataset.select(val_indices.tolist()),
})

print(f"Train rows: {len(dataset['train'])} | Validation rows: {len(dataset['validation'])}")

dataset["train"] = dataset["train"].map(lambda x: {"target_class": int(x["labels"])})
dataset["validation"] = dataset["validation"].map(lambda x: {"target_class": int(x["labels"])})

_split_labels = {"train": np.bincount(np.asarray(dataset["train"]["target_class"], dtype=np.int64), minlength=3),
                 "validation": np.bincount(np.asarray(dataset["validation"]["target_class"], dtype=np.int64), minlength=3)}
print("Split class distribution (none / moderate / severe):")
for _split, _counts in _split_labels.items():
    print(f"  {_split:>10}: none={_counts[0]}  moderate={_counts[1]}  severe={_counts[2]}  (total={_counts.sum()})")

train_plus_val_counts = _split_labels["train"] + _split_labels["validation"]
is_split_consistent = bool(np.array_equal(train_plus_val_counts.astype(int), train_class_counts_full.astype(int)))
print(
    "Train+validation matches full pre-split train distribution: "
    f"{is_split_consistent} | combined={train_plus_val_counts.astype(int).tolist()}"
)

train_classes = np.asarray(dataset["train"]["target_class"], dtype=np.int64)
class_weights_np, class_counts = build_class_weights(train_classes)
print("Weighted loss enabled (effective-number weighted cross entropy)")
print(f"  Class counts [0,1,2]: {class_counts.tolist()}")
print(f"  Class weights [0,1,2]: {[round(float(w), 4) for w in class_weights_np.tolist()]}")
print(f"  EFN beta: {CLASS_BALANCE_BETA} | EFN power: {EFN_WEIGHT_POWER}")


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Custom Models
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class CustomBERTModel(nn.Module):
    """Chunk-level <s> encoding + mean pooling + 3-class depression head."""

    def __init__(
        self,
        pretrained_model_name,
        class_weights,
        chunk_micro_batch_size=16,
        freeze_bottom_layers=FREEZE_BOTTOM_LAYERS,
        use_gradient_checkpointing=False,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.config = self.encoder.config
        self.config.num_labels = NUM_CLASSES
        self.chunk_micro_batch_size = chunk_micro_batch_size
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.classifier = nn.Linear(self.config.hidden_size, NUM_CLASSES)
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))

        if freeze_bottom_layers > 0:
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
            if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
                total_layers = len(self.encoder.encoder.layer)
                n_freeze = min(int(freeze_bottom_layers), total_layers)
                for layer_idx in range(n_freeze):
                    for param in self.encoder.encoder.layer[layer_idx].parameters():
                        param.requires_grad = False
                print(f"Encoder freezing: embeddings + bottom {n_freeze}/{total_layers} layers frozen")
            else:
                print("Encoder freezing skipped: unexpected encoder structure")

        if use_gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

    def _encode_chunks_sequential(self, input_ids, attention_mask):
        """Encode chunk batches sequentially to reduce memory usage."""
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

    def _classification_loss(self, logits, labels):
        return torch.nn.functional.cross_entropy(
            logits,
            labels.long(),
            weight=self.class_weights,
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None, subject_chunk_counts=None):
        loss = None

        if subject_chunk_counts is not None:
            subject_logits_list = []
            subject_labels_list = []
            chunk_idx = 0

            for subject_idx, num_chunks in enumerate(subject_chunk_counts):
                subj_input_ids = input_ids[chunk_idx: chunk_idx + num_chunks]
                subj_attention = attention_mask[chunk_idx: chunk_idx + num_chunks]
                cls_embeddings = self._encode_chunks_sequential(subj_input_ids, subj_attention)

                pooled_cls = torch.mean(cls_embeddings, dim=0)
                subject_logit = self.classifier(pooled_cls)
                subject_logits_list.append(subject_logit)

                if labels is not None:
                    subject_labels_list.append(labels[subject_idx])

                chunk_idx += num_chunks

            subject_logits = torch.stack(subject_logits_list)

            if labels is not None:
                subject_labels = torch.stack(subject_labels_list).long()
                loss = self._classification_loss(subject_logits, subject_labels)
                return {
                    "loss": loss,
                    "logits": subject_logits,
                }

            return {
                "logits": subject_logits,
            }

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = self.dropout(outputs.last_hidden_state[:, 0, :])
        chunk_logits = self.classifier(cls_embedding)
        if labels is not None:
            labels = labels.long()
            loss = self._classification_loss(chunk_logits, labels)
            return {
                "loss": loss,
                "logits": chunk_logits,
            }
        return {
            "logits": chunk_logits,
        }


# Custom collator to handle chunking and subject-level aggregation.
class SubjectChunkingCollator:
    """Chunk each subject and keep chunk counts for subject-level aggregation."""

    def __init__(self, tokenizer, max_len=MAX_LEN, stride=STRIDE):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.stride = stride

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
# Metrics and Prediction Helpers
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def extract_pred_classes(predictions):
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]
    logits = np.asarray(predictions, dtype=float)
    if logits.ndim == 1:
        logits = logits.reshape(-1, NUM_CLASSES)
    return np.argmax(logits, axis=1).astype(np.int64)


def extract_pred_probabilities(predictions):
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]
    logits = np.asarray(predictions, dtype=float)
    if logits.ndim == 1:
        logits = logits.reshape(-1, NUM_CLASSES)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.clip(np.sum(exp_logits, axis=1, keepdims=True), 1e-8, None)
    return probs


def compute_metrics(eval_pred):
    """Trainer metrics for model selection under 3-class training."""
    predictions, true_classes = eval_pred
    pred_classes = extract_pred_classes(predictions)
    true_classes = np.asarray(true_classes, dtype=np.int64).reshape(-1)
    class_accuracy = float(np.mean(pred_classes == true_classes)) if true_classes.size else 0.0
    bal_accuracy = balanced_accuracy(true_classes, pred_classes) if true_classes.size else 0.0
    macro_f1 = float(f1_score(true_classes, pred_classes, average="macro", zero_division=0)) if true_classes.size else 0.0
    return {"class_accuracy": class_accuracy, "balanced_accuracy": bal_accuracy, "macro_f1": macro_f1}


def _model_dir_has_weights(model_dir):
    if not os.path.isdir(model_dir):
        return False
    has_config = os.path.isfile(os.path.join(model_dir, "config.json"))
    has_weights = (
        os.path.isfile(os.path.join(model_dir, "model.safetensors"))
        or os.path.isfile(os.path.join(model_dir, "pytorch_model.bin"))
    )
    return has_config and has_weights


def _load_checkpoint_into_model(model, checkpoint_dir):
    safe_path = os.path.join(checkpoint_dir, "model.safetensors")
    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.isfile(safe_path):
        state_dict = load_safetensors(safe_path)
    elif os.path.isfile(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        return False
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Checkpoint load warning: missing keys={len(missing)}")
    if unexpected:
        print(f"Checkpoint load warning: unexpected keys={len(unexpected)}")
    return True



#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Domain adaptation (DAPT) with short MLM warmup
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def tokenize_mlm(examples):
    texts = [text.lower() if text else "" for text in examples["text"]]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=MAX_LEN)

mlm_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,
    num_train_epochs=1,
    learning_rate=1e-5,
    warmup_steps=3,
    weight_decay=0.03,
    logging_strategy="epoch",
    save_strategy="no",
    fp16=torch.cuda.is_available(),
    seed=SEED,
    data_seed=SEED,
)


domain_model_ready = _model_dir_has_weights(DOMAIN_ADAPTED_MODEL_DIR)
if domain_model_ready:
    print(f"Skipping MLM warmup: found existing domain-adapted model at {DOMAIN_ADAPTED_MODEL_DIR}")
else:
    # Use training text only for MLM warmup (no EPI text usage here).
    daic_text = dataset["train"].remove_columns([c for c in dataset["train"].column_names if c != "text"])
    print(f"samples (DAIC: {len(daic_text)})")

    mlm_tokenized = daic_text.map(
        tokenize_mlm,
        batched=True,
        remove_columns=["text"],
    )

    mlm_model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)
    mlm_model.config.tie_word_embeddings = False
    mlm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    mlm_trainer = Trainer(
        model=mlm_model,
        args=mlm_args,
        train_dataset=mlm_tokenized,
        data_collator=mlm_collator,
    )

    mlm_trainer.train()
    mlm_trainer.save_model(DOMAIN_ADAPTED_MODEL_DIR)
    print(f"Saved domain-adapted checkpoint to: {DOMAIN_ADAPTED_MODEL_DIR}")

    # Free MLM stage GPU memory before classifier fine-tuning.
    del mlm_trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    del mlm_model
    gc.collect()



#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Training Setup
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Load custom model
custom_model = CustomBERTModel(
    DOMAIN_ADAPTED_MODEL_DIR,
    class_weights=class_weights_np,
    chunk_micro_batch_size=16,
    freeze_bottom_layers=FREEZE_BOTTOM_LAYERS,
    use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
)

trainer = Trainer(
    model=custom_model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=SubjectChunkingCollator(tokenizer, max_len=MAX_LEN, stride=STRIDE),
    compute_metrics=compute_metrics,
)

loaded_checkpoint = None
did_train_classifier = False
if os.path.isdir(BEST_MODEL_DIR):
    if _load_checkpoint_into_model(trainer.model, BEST_MODEL_DIR):
        loaded_checkpoint = BEST_MODEL_DIR
        print(f"Skipping classifier training: loaded best_model from {BEST_MODEL_DIR}")
    else:
        print("Best model found but load failed; running classifier training.")
        trainer.train(resume_from_checkpoint=False)
        did_train_classifier = True
else:
    print("No best_model found; training classifier.")
    trainer.train(resume_from_checkpoint=False)
    did_train_classifier = True

# Save training/eval loss curves for run diagnostics.
os.makedirs(PERFORMANCE_DIR, exist_ok=True)
history = trainer.state.log_history if did_train_classifier else []
train_epochs = [x["epoch"] for x in history if "loss" in x and "eval_loss" not in x]
train_losses = [x["loss"] for x in history if "loss" in x and "eval_loss" not in x]
eval_epochs = [x["epoch"] for x in history if "eval_loss" in x]
eval_losses = [x["eval_loss"] for x in history if "eval_loss" in x]

if train_epochs and SAVE_PLOTS:
    pass


# Validation evaluation (subject-level, pure classification metrics)
val_predictions = trainer.predict(dataset["validation"])
val_pred_classes = extract_pred_classes(val_predictions.predictions)
val_pred_probs = extract_pred_probabilities(val_predictions.predictions)
val_true_classes = np.asarray(dataset["validation"]["target_class"], dtype=np.int64)
val_row_ids = extract_exact_ids(dataset["validation"], preferred_columns=("id", "session_id", "csv_id"))

val_acc = float(np.mean(val_pred_classes == val_true_classes))
val_bal_acc = balanced_accuracy(val_true_classes, val_pred_classes)
val_macro_f1 = float(f1_score(val_true_classes, val_pred_classes, average="macro", zero_division=0))

# Report best model used
if loaded_checkpoint:
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    print(f"Loaded best_model: {loaded_checkpoint}")
else:
    best_ckpt = trainer.state.best_model_checkpoint
    best_step = "unknown"
    if best_ckpt is not None:
        ckpt_name = os.path.basename(best_ckpt)
        if ckpt_name.startswith("checkpoint-"):
            best_step = ckpt_name.split("-")[-1]
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    print(f"Best Step: {best_step}")
    print(f"Best Model: {best_ckpt}")

print("\n" + "="*60)
print("VALIDATION RESULTS")
print("="*60)
print(f"Accuracy:          {val_acc:.4f}")
print(f"Balanced Accuracy: {val_bal_acc:.4f}")
print(f"Macro F1:          {val_macro_f1:.4f}")

val_cm = confusion_matrix(val_true_classes, val_pred_classes, labels=[0, 1, 2])
print("Confusion Matrix (rows=true, cols=pred):")
print(val_cm)

write_subject_predictions_csv(
    VAL_PER_SUBJECT_PREDICTIONS_CSV_PATH,
    row_ids=val_row_ids,
    true_classes=val_true_classes,
    pred_classes=val_pred_classes,
    pred_probs=val_pred_probs,
)
print(f"Saved validation per-subject prediction CSV: {VAL_PER_SUBJECT_PREDICTIONS_CSV_PATH}")

# Save overall validation metrics
val_overall_metrics_path = os.path.join(OUTPUT_DIR, "validation_overall_metrics.csv")
write_overall_metrics_csv(
    val_overall_metrics_path,
    {
        "n": len(val_true_classes),
        "accuracy": val_acc,
        "balanced_accuracy": val_bal_acc,
        "macro_f1": val_macro_f1,
    },
)
print(f"Saved validation overall metrics CSV: {val_overall_metrics_path}")

if SAVE_PLOTS:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(val_cm, cmap="Blues")
    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["none", "moderate", "severe"])
    ax.set_yticklabels(["none", "moderate", "severe"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Validation")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(val_cm[i, j]), ha="center", va="center")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    val_cm_path = os.path.join(PERFORMANCE_DIR, "validation_confusion_matrix.png")
    plt.savefig(val_cm_path, dpi=180)
    plt.close()
    print(f"Saved validation confusion matrix: {val_cm_path}")

os.makedirs(PERFORMANCE_DIR, exist_ok=True)
if did_train_classifier and best_ckpt is not None:
    import shutil
    output_dir = training_args.output_dir
    for name in os.listdir(output_dir):
        ckpt_dir = os.path.join(output_dir, name)
        if name.startswith("checkpoint-") and os.path.isdir(ckpt_dir) and ckpt_dir != best_ckpt:
            shutil.rmtree(ckpt_dir)
            print(f"Deleted: {ckpt_dir}")

# Save the final model properly
print(f"\nSaving model to: {BEST_MODEL_DIR}")

# Save full model state (encoder + classifier + dropout)
import torch
torch.save({
    'encoder_state': trainer.model.encoder.state_dict(),
    'classifier_weight': trainer.model.classifier.weight.data,
    'classifier_bias': trainer.model.classifier.bias.data,
    'dropout_rate': DROPOUT_RATE,
    'num_classes': NUM_CLASSES,
}, os.path.join(BEST_MODEL_DIR, 'model_weights.pt'))

print(f"✓ Model weights saved successfully to: {BEST_MODEL_DIR}")
print("="*60)
