import os
import csv
import random
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from scipy.stats import pearsonr
from safetensors.torch import load_file as load_safetensors


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

# Data loading
TRAIN_FILE = "data_train.csv"
TEST_FILE = "data_test.csv"

# Directories
OUTPUT_DIR = "learning_results"
LOGGING_DIR = os.path.join(OUTPUT_DIR, "logs")
PERFORMANCE_DIR = os.path.join(OUTPUT_DIR, "performance_plots")
OVERALL_METRICS_CSV_PATH = os.path.join(OUTPUT_DIR, "holdout_overall_metrics.csv")
PER_SESSION_METRICS_CSV_PATH = os.path.join(OUTPUT_DIR, "holdout_metrics_by_session.csv")
BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model")

# Model
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
MAX_LEN = 1024

# Hardware
BF16_AVAILABLE = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
SAVE_PLOTS = True


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Configs
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    logging_dir=LOGGING_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    warmup_ratio=0.03,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    gradient_accumulation_steps=2,
    num_train_epochs=6,
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    lr_scheduler_type="cosine",
    fp16=torch.cuda.is_available() and not BF16_AVAILABLE,
    bf16=BF16_AVAILABLE,
    seed=SEED,
    data_seed=SEED,
    report_to=[],
)

# Class and Loss Configuration
NUM_CLASSES = 3
CLASS_BALANCE_BETA = 0.999
EFN_WEIGHT_POWER = 1.0
CLASS_SCORE_ANCHORS = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Functions and Helpers
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_class_weights(target_classes, num_classes=NUM_CLASSES, beta=CLASS_BALANCE_BETA, power=EFN_WEIGHT_POWER):
    """Class-balanced weights using effective number of samples."""
    target_labels = np.asarray(target_classes, dtype=np.int64).reshape(-1)
    counts = np.bincount(target_labels, minlength=num_classes).astype(np.float64)
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


def logits_to_scores(predictions):
    """Convert logits/probabilities to continuous plot score in [0,1]."""
    logits = np.asarray(predictions, dtype=float)
    if logits.ndim == 1:
        logits = logits.reshape(-1, NUM_CLASSES)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.clip(np.sum(exp_logits, axis=1, keepdims=True), 1e-8, None)
    scores = probs @ CLASS_SCORE_ANCHORS
    return np.clip(scores.reshape(-1), 0.0, 1.0)


def extract_pred_classes(predictions):
    """Extract predicted class labels from model predictions."""
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]
    logits = np.asarray(predictions, dtype=float)
    if logits.ndim == 1:
        logits = logits.reshape(-1, NUM_CLASSES)
    return np.argmax(logits, axis=1).astype(np.int64)


def extract_pred_scores(predictions):
    """Extract predicted continuous scores from model predictions."""
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]
    return logits_to_scores(predictions)


def balanced_accuracy(y_true, y_pred):
    """Balanced accuracy on a fixed 3-class label space [0,1,2]."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).astype(float)
    recalls = []
    for i in range(3):
        denom = cm[i, :].sum()
        recalls.append((cm[i, i] / denom) if denom > 0 else 0.0)
    return float(np.mean(recalls))


def compute_metrics(eval_pred):
    """Compute evaluation metrics: class_accuracy, balanced_accuracy, macro_f1."""
    predictions, true_classes = eval_pred
    pred_classes = extract_pred_classes(predictions)
    true_classes = np.asarray(true_classes, dtype=np.int64).reshape(-1)
    class_accuracy = float(np.mean(pred_classes == true_classes)) if true_classes.size else 0.0
    bal_accuracy = balanced_accuracy(true_classes, pred_classes) if true_classes.size else 0.0
    macro_f1 = float(f1_score(true_classes, pred_classes, average="macro", zero_division=0)) if true_classes.size else 0.0
    return {"class_accuracy": class_accuracy, "balanced_accuracy": bal_accuracy, "macro_f1": macro_f1}


def write_overall_metrics_csv(output_path, metrics_row):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "rho_pearson",
                "p_value",
                "confusion_matrix",
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
        rows.append(
            {
                "session": session_label,
                "n_samples": int(y_true.size),
                "accuracy": float(np.mean(y_pred == y_true)),
                "balanced_accuracy": balanced_accuracy(y_true, y_pred),
                "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
            }
        )

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["session", "n_samples", "accuracy", "balanced_accuracy", "macro_f1"],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_subject_and_session_ids(session_ids):
    subject_ids = []
    session_labels = []
    for raw_id in session_ids:
        raw_id = str(raw_id)
        if "-" in raw_id:
            subject_id, session_label = raw_id.rsplit("-", 1)
            session_label = f"-{session_label}"
        else:
            subject_id = raw_id
            session_label = "unknown"
        subject_ids.append(subject_id)
        session_labels.append(session_label)
    return subject_ids, session_labels


def parse_subject_and_session_numbers(session_ids):
    subject_ids = []
    session_numbers = []
    for raw_id in session_ids:
        raw_id = str(raw_id)
        if "-" in raw_id:
            subject_id, session_str = raw_id.rsplit("-", 1)
            try:
                session_num = int(session_str)
            except ValueError:
                session_num = None
        else:
            subject_id = raw_id
            session_num = None
        subject_ids.append(subject_id)
        session_numbers.append(session_num)
    return subject_ids, session_numbers


def significance_stars(p_value):
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def save_session_grouped_epi_scatter(x_values, y_values, groups, output_path, title, x_label):
    unique_groups = sorted(set(groups))
    cmap = plt.get_cmap("tab10", max(len(unique_groups), 1))
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    x_range = float(np.ptp(x_values)) if len(x_values) else 0.0
    y_range = float(np.ptp(y_values)) if len(y_values) else 0.0

    fig, ax = plt.subplots(figsize=(11, 7))
    for idx, group in enumerate(unique_groups):
        mask = np.asarray([g == group for g in groups], dtype=bool)
        color = cmap(idx)
        group_x = x_values[mask]
        group_y = y_values[mask]

        mean_x = float(np.mean(group_x))
        mean_y = float(np.mean(group_y))
        spread_x = float(np.std(group_x))
        spread_y = float(np.std(group_y))
        ellipse_width = max(2.0 * spread_x, 0.06 * max(x_range, 1e-6))
        ellipse_height = max(2.0 * spread_y, 0.08 * max(y_range, 1.0))

        ellipse = Ellipse(
            (mean_x, mean_y),
            width=ellipse_width,
            height=ellipse_height,
            facecolor=color,
            edgecolor=color,
            linewidth=1.2,
            linestyle="--",
            alpha=0.14,
            zorder=1,
        )
        ax.add_patch(ellipse)

        ax.scatter(
            group_x,
            group_y,
            alpha=0.82,
            s=46,
            color=color,
            edgecolors="white",
            linewidths=0.6,
            label=group,
            zorder=2,
        )
        ax.scatter(
            [mean_x],
            [mean_y],
            s=110,
            color=color,
            edgecolors="black",
            linewidths=0.9,
            zorder=3,
        )
        ax.annotate(
            group,
            (mean_x, mean_y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            weight="bold",
            color=color,
            zorder=4,
        )

    if len(x_values) >= 2 and float(np.ptp(x_values)) > 0.0:
        fit = np.polyfit(x_values, y_values, 1)
        x_line = np.linspace(float(np.min(x_values)), float(np.max(x_values)), 100)
        y_line = fit[0] * x_line + fit[1]
        ax.plot(x_line, y_line, linestyle="--", color="black", linewidth=1.5, alpha=0.8)

    ax.set_xlabel(x_label)
    ax.set_ylabel("True HAM-D sum")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(
        title="Session",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_session_cloud_grid(x_values, y_values, groups, output_path, title, x_label):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    groups = list(groups)
    unique_groups = sorted(set(groups))
    if not unique_groups:
        return 0

    n_panels = len(unique_groups)
    ncols = min(4, n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    x_min = float(np.min(x_values)) if x_values.size else 0.0
    x_max = float(np.max(x_values)) if x_values.size else 1.0
    y_min = float(np.min(y_values)) if y_values.size else 0.0
    y_max = float(np.max(y_values)) if y_values.size else 1.0

    for plot_i, group in enumerate(unique_groups):
        r, c = divmod(plot_i, ncols)
        ax = axes[r][c]
        mask = np.asarray([g == group for g in groups], dtype=bool)
        gx = x_values[mask]
        gy = y_values[mask]

        ax.scatter(gx, gy, alpha=0.78, s=28, color="#2F6DB0", edgecolors="white", linewidths=0.4)
        if gx.size >= 2 and float(np.ptp(gx)) > 0.0:
            fit = np.polyfit(gx, gy, 1)
            x_line = np.linspace(float(np.min(gx)), float(np.max(gx)), 100)
            y_line = fit[0] * x_line + fit[1]
            ax.plot(x_line, y_line, linestyle="--", color="black", linewidth=1.0, alpha=0.75)

        ax.set_title(f"Session {group} (n={gx.size})", fontsize=9)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(alpha=0.2)
        if r == nrows - 1:
            ax.set_xlabel(x_label)
        if c == 0:
            ax.set_ylabel("True HAM-D sum")

    for j in range(n_panels, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return n_panels


def save_subject_trajectory_grid(session_ids, pred_scores, hamd_values, output_path):
    pred_scores = np.asarray(pred_scores, dtype=float)
    hamd_values = np.asarray(hamd_values, dtype=float)
    subject_ids, session_numbers = parse_subject_and_session_numbers(session_ids)
    subject_ids = np.asarray(subject_ids)
    session_numbers = np.asarray(session_numbers, dtype=object)

    candidates = []
    for subj in sorted(set(subject_ids.tolist())):
        idx = np.where(subject_ids == subj)[0]
        sess = session_numbers[idx]
        valid = [x for x in sess if isinstance(x, int)]
        if len(valid) >= 2 and len(set(valid)) >= 2:
            candidates.append((subj, idx))

    if not candidates:
        return 0

    n_subj = len(candidates)
    ncols = min(4, n_subj)
    nrows = int(np.ceil(n_subj / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), squeeze=False)

    for plot_i, (subj, idx) in enumerate(candidates):
        r, c = divmod(plot_i, ncols)
        ax = axes[r][c]

        sess_nums = session_numbers[idx]
        order = np.argsort([s if isinstance(s, int) else 10**9 for s in sess_nums])
        idx_sorted = idx[order]

        x = np.asarray([int(session_numbers[i]) for i in idx_sorted], dtype=int)
        y_true = hamd_values[idx_sorted]
        y_pred_score = pred_scores[idx_sorted]

        ax.plot(x, y_true, marker="o", color="#1F7FE6", linewidth=1.8, label="True HAM-D")
        ax2 = ax.twinx()
        ax2.plot(x, y_pred_score, marker="s", color="#D17A22", linestyle="--", linewidth=1.8, label="Pred score (0-1)")

        if float(np.ptp(y_pred_score)) == 0.0 or float(np.ptp(y_true)) == 0.0:
            rho = 0.0
            p_val = 1.0
        else:
            rho_res = pearsonr(y_pred_score, y_true)
            rho = float(rho_res.statistic) if rho_res is not None else np.nan
            p_val = float(rho_res.pvalue) if rho_res is not None else np.nan

        if np.isnan(rho):
            rho = 0.0
        if np.isnan(p_val):
            p_val = 1.0

        stars = significance_stars(p_val)
        ax.set_title(f"{subj} | rho={rho:.2f} ({stars})", fontsize=9)
        ax.set_xticks(x.tolist())
        ax.set_xlabel("Session")
        ax.set_ylabel("True HAM-D")

        span_true = float(np.ptp(y_true))
        pad_true = max(1.0, 0.15 * span_true)
        y_true_low = max(0.0, float(np.min(y_true)) - pad_true)
        y_true_high = float(np.max(y_true)) + pad_true
        if y_true_high - y_true_low < 2.0:
            center_true = 0.5 * (float(np.min(y_true)) + float(np.max(y_true)))
            y_true_low = max(0.0, center_true - 1.0)
            y_true_high = center_true + 1.0
        if y_true_high <= y_true_low:
            y_true_low, y_true_high = 0.0, 1.0
        ax.set_ylim(y_true_low, y_true_high)

        ax2.set_ylabel("Pred score (0-1)")
        span_pred = float(np.ptp(y_pred_score))
        pad_pred = max(0.02, 0.2 * span_pred)
        y_pred_low = max(0.0, float(np.min(y_pred_score)) - pad_pred)
        y_pred_high = min(1.0, float(np.max(y_pred_score)) + pad_pred)
        if y_pred_high - y_pred_low < 0.08:
            center_pred = 0.5 * (float(np.min(y_pred_score)) + float(np.max(y_pred_score)))
            y_pred_low = max(0.0, center_pred - 0.04)
            y_pred_high = min(1.0, center_pred + 0.04)
        if y_pred_high <= y_pred_low:
            y_pred_low, y_pred_high = 0.0, 1.0
        ax2.set_ylim(y_pred_low, y_pred_high)

        ax.grid(alpha=0.2)
        if plot_i == 0:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")

    for j in range(n_subj, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    fig.suptitle("Subject Trajectories (subjects with >=2 sessions) | dual y-axis with per-panel zoom", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return n_subj


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss = torch.nn.functional.cross_entropy(logits, labels.long(), weight=weights)
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels.long())

        return (loss, outputs) if return_outputs else loss


def _load_checkpoint_into_model(model, checkpoint_dir):
    """Load model weights from checkpoint directory (safetensors or pytorch_model.bin)."""
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
# Dataset loading and splitting
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def main():
    set_seed(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)
    os.makedirs(PERFORMANCE_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    # Load and split data
    train_dataset = load_dataset("csv", data_files={"train": TRAIN_FILE})["train"]
    test_dataset = load_dataset("csv", data_files={"test": TEST_FILE}, sep=";", encoding="utf-8")["test"]

    train_dataset = train_dataset.select_columns(["text", "labels"])
    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    train_indices, val_indices = train_test_split(
        np.arange(len(dataset["train"])),
        test_size=0.2,
        stratify=np.asarray(dataset["train"]["labels"], dtype=np.int64),
        random_state=SEED,
    )

    dataset = DatasetDict(
        {
            "train": dataset["train"].select(train_indices.tolist()),
            "validation": dataset["train"].select(val_indices.tolist()),
            "test": dataset["test"],
        }
    )

    print(f"Train rows: {len(dataset['train'])} | Validation rows: {len(dataset['validation'])} | Test rows: {len(dataset['test'])}")

    dataset["train"] = dataset["train"].map(lambda x: {"target_class": int(x["labels"])})
    dataset["validation"] = dataset["validation"].map(lambda x: {"target_class": int(x["labels"])})

    # Compute class weights
    train_classes = np.asarray(dataset["train"]["target_class"], dtype=np.int64)
    class_weights_np, class_counts = build_class_weights(train_classes)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32)

    print("Weighted loss enabled (effective-number weighted cross entropy)")
    print(f"  Class counts [0,1,2]: {class_counts.tolist()}")
    print(f"  Class weights [0,1,2]: {[round(float(w), 4) for w in class_weights_np.tolist()]}")
    print(f"  Model: {BASE_MODEL}")

    def _normalize_text_batch(text_values):
        normalized = []
        for value in text_values:
            if value is None:
                normalized.append("")
            elif isinstance(value, float) and np.isnan(value):
                normalized.append("")
            else:
                normalized.append(str(value))
        return normalized

    # Tokenize datasets
    def tokenize_train_val(batch):
        texts = _normalize_text_batch(batch["text"])
        enc = tokenizer(texts, truncation=True, max_length=MAX_LEN)
        enc["labels"] = [int(x) for x in batch["target_class"]]
        return enc

    def tokenize_test(batch):
        texts = _normalize_text_batch(batch["text"])
        return tokenizer(texts, truncation=True, max_length=MAX_LEN)

    train_tok = dataset["train"].map(tokenize_train_val, batched=True)
    val_tok = dataset["validation"].map(tokenize_train_val, batched=True)
    test_tok = dataset["test"].map(tokenize_test, batched=True)

    keep_cols_train = ["input_ids", "attention_mask", "labels"]
    keep_cols_test = ["input_ids", "attention_mask"]

    train_tok = train_tok.remove_columns([c for c in train_tok.column_names if c not in keep_cols_train])
    val_tok = val_tok.remove_columns([c for c in val_tok.column_names if c not in keep_cols_train])
    test_tok = test_tok.remove_columns([c for c in test_tok.column_names if c not in keep_cols_test])


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Training Setup
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Model initialization
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_CLASSES,
        problem_type="single_label_classification",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # Trainer initialization
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    # Skip training if best model already exists
    loaded_checkpoint = None
    did_train = False
    if os.path.isdir(BEST_MODEL_DIR):
        if _load_checkpoint_into_model(trainer.model, BEST_MODEL_DIR):
            loaded_checkpoint = BEST_MODEL_DIR
            print(f"Skipping training: loaded best_model from {BEST_MODEL_DIR}")
        else:
            print("Best model found but load failed; running training.")
            trainer.train(resume_from_checkpoint=False)
            did_train = True
    else:
        print("No best_model found; training...")
        trainer.train(resume_from_checkpoint=False)
        did_train = True

    # Only save and update checkpoints if we actually trained
    if did_train:
        best_ckpt = trainer.state.best_model_checkpoint
        print("\n" + "=" * 60)
        print("TRAINING RESULTS")
        print("=" * 60)
        print(f"Best Model Checkpoint: {best_ckpt}")

        trainer.save_model(BEST_MODEL_DIR)
        tokenizer.save_pretrained(BEST_MODEL_DIR)
        print(f"Saved best model to: {BEST_MODEL_DIR}")

        if best_ckpt is not None:
            for name in os.listdir(OUTPUT_DIR):
                ckpt_dir = os.path.join(OUTPUT_DIR, name)
                if name.startswith("checkpoint-") and os.path.isdir(ckpt_dir) and ckpt_dir != best_ckpt:
                    shutil.rmtree(ckpt_dir)
    else:
        print("\n" + "=" * 60)
        print("SKIPPED TRAINING")
        print("=" * 60)
        print(f"Using existing best model from: {loaded_checkpoint}")


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Evaluation on Holdout EPI data
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Validation evaluation
    val_predictions = trainer.predict(val_tok)
    val_pred_classes = extract_pred_classes(val_predictions.predictions)
    val_true_classes = np.asarray(dataset["validation"]["target_class"], dtype=np.int64)

    val_acc = float(np.mean(val_pred_classes == val_true_classes))
    val_bal_acc = balanced_accuracy(val_true_classes, val_pred_classes)
    val_macro_f1 = float(f1_score(val_true_classes, val_pred_classes, average="macro", zero_division=0))

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:          {val_acc:.4f}")
    print(f"Balanced Accuracy: {val_bal_acc:.4f}")
    print(f"Macro F1:          {val_macro_f1:.4f}")

    val_cm = confusion_matrix(val_true_classes, val_pred_classes, labels=[0, 1, 2])
    print("Confusion Matrix (rows=true, cols=pred):")
    print(val_cm)

    if SAVE_PLOTS:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(val_cm, cmap="Blues")
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(["none", "moderate", "severe"])
        ax.set_yticklabels(["none", "moderate", "severe"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Validation Confusion Matrix")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(val_cm[i, j]), ha="center", va="center")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        val_cm_path = os.path.join(PERFORMANCE_DIR, "val_confusion_matrix.png")
        plt.savefig(val_cm_path, dpi=180)
        plt.close()
        print(f"Saved validation confusion matrix: {val_cm_path}")

    predictions = trainer.predict(test_tok)
    test_scores = extract_pred_scores(predictions.predictions)
    test_pred_classes = extract_pred_classes(predictions.predictions)

    if "hamd_sum" in dataset["test"].column_names:
        hamd_true = np.asarray(dataset["test"]["hamd_sum"], dtype=float)
        hamd_true_classes = np.where(hamd_true <= 7, 0, np.where(hamd_true <= 23, 1, 2)).astype(np.int64)

        test_acc = float(np.mean(test_pred_classes == hamd_true_classes))
        test_bal_acc = balanced_accuracy(hamd_true_classes, test_pred_classes)
        test_macro_f1 = float(f1_score(hamd_true_classes, test_pred_classes, average="macro", zero_division=0))

        pearson_test = pearsonr(test_scores, hamd_true)
        rho_hamd = float(pearson_test.statistic)
        p_hamd = float(pearson_test.pvalue)
        if np.isnan(rho_hamd):
            rho_hamd = 0.0
        if np.isnan(p_hamd):
            p_hamd = 1.0

        print("\n" + "=" * 60)
        print("TEST EVALUATION (EPI HOLDOUT)")
        print("=" * 60)
        print(f"Accuracy:          {test_acc:.4f}")
        print(f"Balanced Accuracy: {test_bal_acc:.4f}")
        print(f"Macro F1:          {test_macro_f1:.4f}")
        print(f"Rho (Pearson, continuous vs HAM-D): {rho_hamd:.4f}")
        print(f"p-value:           {p_hamd:.4g}")

        test_cm = confusion_matrix(hamd_true_classes, test_pred_classes, labels=[0, 1, 2])
        print("Confusion Matrix (rows=true, cols=pred):")
        print(test_cm)

        if SAVE_PLOTS:
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(test_cm, cmap="Blues")
            ax.set_xticks([0, 1, 2])
            ax.set_yticks([0, 1, 2])
            ax.set_xticklabels(["none", "moderate", "severe"])
            ax.set_yticklabels(["none", "moderate", "severe"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Test Confusion Matrix (EPI)")
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, str(test_cm[i, j]), ha="center", va="center")
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            test_cm_path = os.path.join(PERFORMANCE_DIR, "test_confusion_matrix.png")
            plt.savefig(test_cm_path, dpi=180)
            plt.close()
            print(f"Saved test confusion matrix: {test_cm_path}")

        write_overall_metrics_csv(
            OVERALL_METRICS_CSV_PATH,
            {
                "accuracy": test_acc,
                "balanced_accuracy": test_bal_acc,
                "macro_f1": test_macro_f1,
                "rho_pearson": rho_hamd,
                "p_value": p_hamd,
                "confusion_matrix": str(test_cm.tolist()),
            },
        )
        print(f"Saved overall metrics CSV: {OVERALL_METRICS_CSV_PATH}")

        if "session_id" in dataset["test"].column_names:
            session_ids = dataset["test"]["session_id"]
            _, session_labels = parse_subject_and_session_ids(session_ids)

            write_per_session_metrics_csv(
                PER_SESSION_METRICS_CSV_PATH,
                session_labels=session_labels,
                true_classes=hamd_true_classes,
                pred_classes=test_pred_classes,
            )
            print(f"Saved per-session metrics CSV: {PER_SESSION_METRICS_CSV_PATH}")

            if SAVE_PLOTS:
                session_plot_path = os.path.join(PERFORMANCE_DIR, "hamd_scatter_by_session.png")
                save_session_grouped_epi_scatter(
                    x_values=np.asarray(test_scores, dtype=float),
                    y_values=np.asarray(hamd_true, dtype=float),
                    groups=session_labels,
                    output_path=session_plot_path,
                    title=f"Predicted continuous score (0-1) vs True HAM-D by session  (rho={rho_hamd:.3f}, p={p_hamd:.3g})",
                    x_label="Predicted continuous score (0-1)",
                )
                print(f"Saved session scatter plot: {session_plot_path}")

                session_grid_path = os.path.join(PERFORMANCE_DIR, "hamd_cloud_grid_by_session.png")
                n_session_panels = save_session_cloud_grid(
                    x_values=np.asarray(test_scores, dtype=float),
                    y_values=np.asarray(hamd_true, dtype=float),
                    groups=session_labels,
                    output_path=session_grid_path,
                    title="Predicted continuous score (0-1) vs True HAM-D per session",
                    x_label="Predicted continuous score (0-1)",
                )
                if n_session_panels > 0:
                    print(f"Saved session cloud grid ({n_session_panels} panels): {session_grid_path}")

                trajectory_grid_path = os.path.join(PERFORMANCE_DIR, "subject_trajectories_grid.png")
                n_traj_subjects = save_subject_trajectory_grid(
                    session_ids=session_ids,
                    pred_scores=np.asarray(test_scores, dtype=float),
                    hamd_values=np.asarray(hamd_true, dtype=float),
                    output_path=trajectory_grid_path,
                )
                if n_traj_subjects > 0:
                    print(f"Saved subject trajectory grid ({n_traj_subjects} subjects): {trajectory_grid_path}")
                else:
                    print("Skipped trajectory grid: no subjects with >=2 valid sessions in current test file.")
    else:
        print("\nHAM-D column not found in test set; skipped test evaluation.")


if __name__ == "__main__":
    main()
    main()
