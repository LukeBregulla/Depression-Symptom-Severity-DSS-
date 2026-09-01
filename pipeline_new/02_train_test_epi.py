import os
import gc
import csv
import re
import shutil
import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, auc, average_precision_score, precision_recall_curve


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


# Edit stage paths here.
PROJECT_ROOT = "/zi/home/luke.bregulla/Desktop/DSS"
EPI_FILE = os.path.join(PROJECT_ROOT, "data/data_epi.csv")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results/02_epi")
TEXT_INPUT = "full_transcript"
OUTPUT_DIR = RESULTS_ROOT
BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model")
LOGGING_DIR = os.path.join(OUTPUT_DIR, "logs")
PLOT_DIR = os.path.join(OUTPUT_DIR, "performance_plots")
VALIDATION_METRICS_CSV = os.path.join(OUTPUT_DIR, "validation_metrics.csv")
SUBJECT_PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "subject_predictions.csv")
SEVERE_METRICS_CSV = os.path.join(OUTPUT_DIR, "severe_discrimination_metrics.csv")
DATA_DISTRIBUTION_PLOT_PATH = os.path.join(PLOT_DIR, "dataset_class_distribution_train_test.png")

os.environ["TENSORBOARD_LOGGING_DIR"] = LOGGING_DIR

# Load tokenizer
BASE_MODEL = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
SAVE_PLOTS = True


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Configs
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# One "sample" is a whole subject, i.e. 30-100 chunks of 512 tokens for the EPIsoDE sessions.
# Use 8 subjects per batch for stable gradient dynamics and better batch statistics (critical for performance)
SUBJECT_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
CHUNK_MICRO_BATCH_SIZE = 8

MAX_LEN = 512
STRIDE = 256  
USE_GRADIENT_CHECKPOINTING = True
FREEZE_BOTTOM_LAYERS = 4
DROPOUT_RATE = 0.3
NUM_CLASSES = 2
CLASS_BALANCE_BETA = 0.999
EFN_WEIGHT_POWER = 1.0
USE_FP16 = True  # CRITICAL: Must be True for optimal training (disabling causes significant performance degradation)

MAX_EPOCHS = 20
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
FINAL_N_FOLDS = 5


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Permutation Testing Utilities
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Functions and Helpers
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def reseed():
    """Re-seed all RNGs for fold reproducibility."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

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
    """Balanced accuracy on the fixed binary label space [0,1]."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
    recalls = []
    for i in range(2):
        denom = cm[i, :].sum()
        recalls.append((cm[i, i] / denom) if denom > 0 else 0.0)
    return float(np.mean(recalls))


def write_single_row_csv(output_path, row):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_rows_csv(output_path, fieldnames, rows):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def prepare_model_text(dataset, dataset_name):
    """All runs use the full transcript view; patient-only mode is intentionally disabled."""
    return dataset


def save_class_distribution_plot(distribution_map, output_path):
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
        fig.suptitle("Class Distribution: Train (left) vs Test (right)")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Dataset loading
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def load_csv_dataset(path, split):
    """Load a CSV whose delimiter may be comma or tab depending on how it was generated."""
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
    sep = "\t" if header.count("\t") > header.count(",") else ","
    return load_dataset("csv", data_files={split: path}, sep=sep, encoding="utf-8")[split]


epi_dataset = load_csv_dataset(EPI_FILE, "train")
if "labels" not in epi_dataset.column_names:
    raise ValueError(
        f"Required 'labels' column not found in {EPI_FILE}. "
        f"Columns read: {epi_dataset.column_names}"
    )

epi_dataset = epi_dataset.map(
    lambda x: {"id": str(x["id"]), "text": x["text"], "labels": int(x["labels"]), "continuous_score": float(x["continuous_score"])},
    remove_columns=epi_dataset.column_names,
)
train_dataset = epi_dataset
# Paper artifact: EPI class distribution.
train_class_counts_full = np.bincount(np.asarray(train_dataset["labels"], dtype=np.int64), minlength=NUM_CLASSES).astype(int)
class_distribution = {"EPI": train_class_counts_full}
save_class_distribution_plot(class_distribution, DATA_DISTRIBUTION_PLOT_PATH)
print(f"Saved dataset class distribution plot: {DATA_DISTRIBUTION_PLOT_PATH}")

train_dataset = prepare_model_text(train_dataset, "training")
train_dataset = train_dataset.select_columns(["id", "text", "labels", "continuous_score"])

dataset = DatasetDict({"train": train_dataset})

train_classes = np.asarray(dataset["train"]["labels"], dtype=np.int64)
train_participants = np.asarray([participant_key(sample_id) for sample_id in dataset["train"]["id"]], dtype=object)
print(f"Training labels [0,1]: {np.bincount(train_classes, minlength=NUM_CLASSES).tolist()}")



#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Custom Models
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class CustomBERTModel(nn.Module):
    """Chunk-level encoding, subject mean pooling, and binary classification head."""

    def __init__(
        self,
        pretrained_model_name,
        class_weights,
        chunk_micro_batch_size=CHUNK_MICRO_BATCH_SIZE,
        freeze_bottom_layers=FREEZE_BOTTOM_LAYERS,
        dropout_rate=DROPOUT_RATE,
        use_gradient_checkpointing=False,
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
            self.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

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
                pooled_cls = pooled_cls.to(self.classifier.weight.dtype)
                subject_logit = self.classifier(pooled_cls)
                subject_logits_list.append(subject_logit)

                if labels is not None:
                    subject_labels_list.append(labels[subject_idx])

                chunk_idx += num_chunks

            subject_logits = torch.stack(subject_logits_list)

            if not torch.isfinite(subject_logits).all():
                raise FloatingPointError("Non-finite subject logits detected before loss computation.")

            if labels is not None:
                subject_labels = torch.stack(subject_labels_list).long()
                loss = self._classification_loss(subject_logits, subject_labels)
                return SequenceClassifierOutput(loss=loss, logits=subject_logits)

            return SequenceClassifierOutput(logits=subject_logits)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = self.dropout(outputs.last_hidden_state[:, 0, :])
        cls_embedding = cls_embedding.to(self.classifier.weight.dtype)
        chunk_logits = self.classifier(cls_embedding)
        if not torch.isfinite(chunk_logits).all():
            raise FloatingPointError("Non-finite chunk logits detected before loss computation.")
        if labels is not None:
            labels = labels.long()
            loss = self._classification_loss(chunk_logits, labels)
            return SequenceClassifierOutput(loss=loss, logits=chunk_logits)
        return SequenceClassifierOutput(logits=chunk_logits)


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


def extract_pred_and_prob(predictions):
    """Extract predicted classes and probability of severe class."""
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]
    logits = np.asarray(predictions, dtype=float)
    if logits.ndim == 1:
        logits = logits.reshape(-1, NUM_CLASSES)
    pred_classes = np.argmax(logits, axis=1).astype(np.int64)
    # Softmax to get probabilities
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    prob_severe = probs[:, 1]  # Probability of class 1 (severe)
    return pred_classes, prob_severe



def best_epoch_from_log(log_history):
    """Return the epoch with the highest validation macro F1."""
    best_score, best_epoch = None, None
    for record in log_history:
        if "eval_macro_f1" in record and (best_score is None or record["eval_macro_f1"] > best_score):
            best_score = record["eval_macro_f1"]
            best_epoch = record.get("epoch")
    return int(round(best_epoch)) if best_epoch else None


def compute_metrics(eval_pred):
    """Trainer metrics for model selection under 3-class training."""
    predictions, true_classes = eval_pred
    pred_classes = extract_pred_classes(predictions)
    true_classes = np.asarray(true_classes, dtype=np.int64).reshape(-1)
    class_accuracy = float(np.mean(pred_classes == true_classes)) if true_classes.size else 0.0
    bal_accuracy = balanced_accuracy(true_classes, pred_classes) if true_classes.size else 0.0
    macro_f1 = float(f1_score(true_classes, pred_classes, labels=[0, 1], average="macro", zero_division=0)) if true_classes.size else 0.0
    return {"class_accuracy": class_accuracy, "balanced_accuracy": bal_accuracy, "macro_f1": macro_f1}


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Training Setup
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

splitter = StratifiedGroupKFold(n_splits=FINAL_N_FOLDS, shuffle=True, random_state=SEED)

shutil.rmtree(BEST_MODEL_DIR, ignore_errors=True)
overall_counts = np.bincount(train_classes, minlength=NUM_CLASSES)
if int(overall_counts.min()) < splitter.n_splits:
    print(
        "WARNING: EPI has fewer subjects in at least one class than CV folds; "
        f"class counts are {overall_counts.tolist()} for {splitter.n_splits} folds."
    )


def run_cv(efn_power, weight_decay, dropout_rate, config_dir, n_folds):
    """Run CV for one parameter configuration."""
    if n_folds != FINAL_N_FOLDS:
        raise ValueError(f"n_folds must be {FINAL_N_FOLDS}, got {n_folds}")
    cv_splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    cv_splits = list(cv_splitter.split(np.zeros(len(dataset["train"])), train_classes, groups=train_participants))
    fold_scores, fold_best_epochs = [], []
    fold_predictions = []
    best_fold_score = -np.inf
    # every training subject is predicted exactly once, by a model that never saw it
    oof_true_by_index = np.empty(len(dataset["train"]), dtype=np.int64)
    oof_pred_by_index = np.empty(len(dataset["train"]), dtype=np.int64)
    oof_prob_by_index = np.empty(len(dataset["train"]), dtype=np.float64)

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits, start=1):
        print(f"\n--- Fold {fold_idx}/{n_folds} ---")

        fold_dataset = DatasetDict({
            "train": dataset["train"].select(train_idx.tolist()),
            "validation": dataset["train"].select(val_idx.tolist()),
        })
        fold_dataset["train"] = fold_dataset["train"].map(lambda x: {**x, "target_class": int(x["labels"])})
        fold_dataset["validation"] = fold_dataset["validation"].map(lambda x: {**x, "target_class": int(x["labels"])})

        fold_train_classes = np.asarray(fold_dataset["train"]["target_class"], dtype=np.int64)
        fold_class_weights_np, _ = build_class_weights(fold_train_classes, power=efn_power)

        fold_output_dir = os.path.join(config_dir, f"fold_{fold_idx}")
        fold_args = TrainingArguments(
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

        fold_model = CustomBERTModel(
            BASE_MODEL,
            class_weights=fold_class_weights_np,
            chunk_micro_batch_size=CHUNK_MICRO_BATCH_SIZE,
            freeze_bottom_layers=FREEZE_BOTTOM_LAYERS,
            dropout_rate=dropout_rate,
            use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        )

        fold_trainer = Trainer(
            model=fold_model,
            args=fold_args,
            train_dataset=fold_dataset["train"],
            eval_dataset=fold_dataset["validation"],
            data_collator=SubjectChunkingCollator(tokenizer, max_len=MAX_LEN, stride=STRIDE),
            compute_metrics=compute_metrics,
        )
        # The custom model computes its own loss and does not accept Trainer's
        # optional num_items_in_batch argument.
        fold_trainer.model_accepts_loss_kwargs = False

        fold_trainer.train(resume_from_checkpoint=False)

        fold_prediction_output = fold_trainer.predict(fold_dataset["validation"])
        fold_pred_classes, fold_prob_severe = extract_pred_and_prob(fold_prediction_output.predictions)
        fold_true_classes = np.asarray(fold_dataset["validation"]["target_class"], dtype=np.int64)
        fold_train_prediction_output = fold_trainer.predict(fold_dataset["train"])
        fold_train_pred_classes = extract_pred_classes(fold_train_prediction_output.predictions)
        fold_train_true_classes = np.asarray(fold_dataset["train"]["target_class"], dtype=np.int64)
        fold_train_macro_f1 = float(
            f1_score(
                fold_train_true_classes,
                fold_train_pred_classes,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        )
        fold_macro_f1 = float(f1_score(fold_true_classes, fold_pred_classes, labels=[0, 1], average="macro", zero_division=0))
        fold_scores.append(fold_macro_f1)
        best_fold_score = max(best_fold_score, fold_macro_f1)
        fold_predictions.append({
            "fold": fold_idx,
            "val_indices": val_idx.copy(),
            "true_classes": fold_true_classes.copy(),
            "pred_classes": fold_pred_classes.copy(),
        })
        oof_true_by_index[val_idx] = fold_true_classes
        oof_pred_by_index[val_idx] = fold_pred_classes
        oof_prob_by_index[val_idx] = fold_prob_severe
        fold_best_epochs.append(best_epoch_from_log(fold_trainer.state.log_history))

        print(f"Fold {fold_idx} train macro F1: {fold_train_macro_f1:.4f}")
        print(f"Fold {fold_idx} validation macro F1: {fold_macro_f1:.4f} (best epoch {fold_best_epochs[-1]})")
        print(f"Fold {fold_idx} train labels: {np.bincount(fold_train_true_classes, minlength=NUM_CLASSES).tolist()}")
        print(f"Fold {fold_idx} validation labels: {np.bincount(fold_true_classes, minlength=NUM_CLASSES).tolist()}")
        print(f"Fold {fold_idx} validation predictions: {np.bincount(fold_pred_classes, minlength=NUM_CLASSES).tolist()}")

        # each fold builds its own encoder; without this the folds accumulate on the GPU
        del fold_trainer, fold_model, fold_prediction_output, fold_train_prediction_output
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # fold checkpoints are only needed to pick the best epoch, not afterwards
        shutil.rmtree(fold_output_dir, ignore_errors=True)

    return {
        "fold_scores": fold_scores,
        "mean": float(np.mean(fold_scores)),
        "std": float(np.std(fold_scores)),
        "best_epochs": fold_best_epochs,
        "oof_true": oof_true_by_index,
        "oof_pred": oof_pred_by_index,
        "oof_prob": oof_prob_by_index,
        "fold_predictions": fold_predictions,
    }

def refit_on_full_data(efn_power, weight_decay, dropout_rate, config_dir, num_epochs):
    """Retrain once on 100% of the training data; this is the artifact used for inference."""
    reseed()
    full_ds = dataset["train"].map(lambda x: {**x, "target_class": int(x["labels"])})
    full_classes = np.asarray(full_ds["target_class"], dtype=np.int64)
    class_weights_np, _ = build_class_weights(full_classes, power=efn_power)

    refit_output_dir = os.path.join(config_dir, "refit_full")
    shutil.rmtree(refit_output_dir, ignore_errors=True)

    refit_args = TrainingArguments(
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

    refit_model = CustomBERTModel(
        BASE_MODEL,
        class_weights=class_weights_np,
        chunk_micro_batch_size=CHUNK_MICRO_BATCH_SIZE,
        freeze_bottom_layers=FREEZE_BOTTOM_LAYERS,
        dropout_rate=dropout_rate,
        use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
    )

    refit_trainer = Trainer(
        model=refit_model,
        args=refit_args,
        train_dataset=full_ds,
        data_collator=SubjectChunkingCollator(tokenizer, max_len=MAX_LEN, stride=STRIDE),
    )
    refit_trainer.model_accepts_loss_kwargs = False
    refit_trainer.train(resume_from_checkpoint=False)

    shutil.rmtree(BEST_MODEL_DIR, ignore_errors=True)
    refit_trainer.save_model(BEST_MODEL_DIR)
    tokenizer.save_pretrained(BEST_MODEL_DIR)

    del refit_trainer, refit_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    shutil.rmtree(refit_output_dir, ignore_errors=True)


print(f"\nUsing hardcoded params: efn_power={EFN_WEIGHT_POWER:g} wd={WEIGHT_DECAY:g} dropout={DROPOUT_RATE:g}")
print(f"Running final {FINAL_N_FOLDS}-fold training...\n")

shutil.rmtree(BEST_MODEL_DIR, ignore_errors=True)
final_config_dir = os.path.join(OUTPUT_DIR, "final_config")
shutil.rmtree(final_config_dir, ignore_errors=True)
os.makedirs(final_config_dir, exist_ok=True)

cv_result = run_cv(
    EFN_WEIGHT_POWER,
    WEIGHT_DECAY,
    DROPOUT_RATE,
    final_config_dir,
    FINAL_N_FOLDS,
)

fold_scores = cv_result["fold_scores"]
fold_best_epochs = cv_result["best_epochs"]
oof_true_classes = cv_result["oof_true"]
oof_pred_classes = cv_result["oof_pred"]
oof_prob_severe = cv_result["oof_prob"]
if not fold_scores:
    raise RuntimeError("No fold completed successfully.")
best_fold_position = int(np.argmax(fold_scores))
best_fold_number = best_fold_position + 1

# CV is only the performance estimate; the shipped model is refit on all data.
refit_epochs = int(np.median([e for e in fold_best_epochs if e])) if any(fold_best_epochs) else MAX_EPOCHS
print(f"\nRefitting on full training data for {refit_epochs} epoch(s)...")
refit_on_full_data(EFN_WEIGHT_POWER, WEIGHT_DECAY, DROPOUT_RATE, final_config_dir, refit_epochs)
shutil.rmtree(final_config_dir, ignore_errors=True)
print(f"Saved full-data refit model: {BEST_MODEL_DIR}")


# Subject-level out-of-fold predictions; continuous-score association is done separately downstream.
continuous_scores = np.asarray(dataset["train"]["continuous_score"], dtype=np.float64)
with open(SUBJECT_PREDICTIONS_CSV, "w", newline="", encoding="utf-8") as oof_file:
    oof_writer = csv.writer(oof_file)
    oof_writer.writerow([
        "id", "participant", "true_label", "pred_label",
        "p_mild_moderate", "p_severe", "continuous_score",
    ])
    for row_id, true_class, pred_class, prob_severe, cont in zip(
        dataset["train"]["id"], oof_true_classes, oof_pred_classes, oof_prob_severe, continuous_scores
    ):
        oof_writer.writerow([
            str(row_id), participant_key(row_id), int(true_class), int(pred_class),
            float(1.0 - prob_severe), float(prob_severe), float(cont),
        ])
print(f"Saved subject predictions: {SUBJECT_PREDICTIONS_CSV}")

if not np.array_equal(np.sort(oof_true_classes), np.sort(train_classes)):
    raise RuntimeError("OOF true labels do not cover the training set exactly once.")
if not np.all(np.isfinite(oof_pred_classes)):
    raise RuntimeError("OOF predictions contain non-finite values.")
if not np.all(np.isfinite(oof_prob_severe)):
    raise RuntimeError("OOF probabilities contain non-finite values.")

# Headline metrics are pooled over all out-of-fold predictions, not a single fold.
val_acc = float(np.mean(oof_pred_classes == oof_true_classes))
val_bal_acc = balanced_accuracy(oof_true_classes, oof_pred_classes)
val_macro_f1 = float(f1_score(oof_true_classes, oof_pred_classes, labels=[0, 1], average="macro", zero_division=0))
try:
    val_roc_auc_severe = float(roc_auc_score(oof_true_classes == 1, oof_prob_severe))
    val_average_precision_severe = float(average_precision_score(oof_true_classes == 1, oof_prob_severe))
except ValueError:
    val_roc_auc_severe = np.nan
    val_average_precision_severe = np.nan

# Permute whole participant blocks, not individual EPI sessions.
subject_ids = np.asarray(dataset["train"]["id"], dtype=object)
perm_result = permutation_participant_wise(oof_true_classes, oof_pred_classes, participant_ids=subject_ids, n_permutations=10000, seed=SEED, label_space=[0, 1])
print("\n" + "="*60)
print("CROSS-VALIDATION RESULTS")
print("="*60)
print(f"Fold macro F1: {[round(s, 4) for s in fold_scores]}")
print(f"Cross-validated macro F1: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")
print(f"Parameters: lr={LEARNING_RATE:g} weight_decay={WEIGHT_DECAY:g} class-weight power={EFN_WEIGHT_POWER:g}")
print(f"Best epochs per fold: {fold_best_epochs}")
print(f"Best fold: {best_fold_number} (macro F1={fold_scores[best_fold_position]:.4f})")

print("\n" + "="*60)
print("POOLED OUT-OF-FOLD RESULTS")
print("="*60)
print(f"Accuracy:          {val_acc:.4f}")
print(f"Balanced Accuracy: {val_bal_acc:.4f}")
print(f"Macro F1:          {val_macro_f1:.4f}")
print(f"ROC-AUC severe:          {val_roc_auc_severe:.4f}")
print(f"Average precision severe: {val_average_precision_severe:.4f}")
print(f"Participant-level permutation p-value: {perm_result['perm_p_value']:.4f}")

val_cm = confusion_matrix(oof_true_classes, oof_pred_classes, labels=[0, 1])
print("Confusion Matrix (rows=true, cols=pred):")
print(val_cm)

val_precision = precision_score(oof_true_classes, oof_pred_classes, labels=[0, 1], average=None, zero_division=0)
val_recall = recall_score(oof_true_classes, oof_pred_classes, labels=[0, 1], average=None, zero_division=0)
class_names = ["mild_moderate", "severe"]
print("Per-class Precision / Recall:")
for class_id, name in enumerate(class_names):
    print(f"  {name:<10}  precision={val_precision[class_id]:.4f}  recall={val_recall[class_id]:.4f}")

write_single_row_csv(
    SEVERE_METRICS_CSV,
    {
        "positive_class": "severe",
        "support": int(np.sum(oof_true_classes == 1)),
        "precision": float(val_precision[1]),
        "recall": float(val_recall[1]),
        "roc_auc": val_roc_auc_severe,
        "average_precision": val_average_precision_severe,
    },
)
print(f"Saved severe discrimination metrics: {SEVERE_METRICS_CSV}")

metrics_row = {
    "n": int(len(oof_true_classes)),
    "cv_estimator": "full_cv_oof",
    "n_folds": FINAL_N_FOLDS,
    "accuracy": val_acc,
    "balanced_accuracy": val_bal_acc,
    "macro_f1": val_macro_f1,
    "roc_auc_severe": val_roc_auc_severe,
    "average_precision_severe": val_average_precision_severe,
    "support_mild_moderate": int(np.sum(oof_true_classes == 0)),
    "support_severe": int(np.sum(oof_true_classes == 1)),
    **{f"fold_{i}_macro_f1": s for i, s in enumerate(fold_scores, start=1)},
    "best_fold": best_fold_number,
    "best_fold_macro_f1": float(np.max(fold_scores)),
    "cv_macro_f1_mean": float(np.mean(fold_scores)),
    "cv_macro_f1_std": float(np.std(fold_scores)),
    "perm_observed_macro_f1": perm_result["observed_macro_f1"],
    "perm_null_macro_f1_mean": perm_result["null_macro_f1_mean"],
    "perm_p_value": perm_result["perm_p_value"],
    "perm_n_permutations": perm_result["n_permutations"],
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "freeze_bottom_layers": FREEZE_BOTTOM_LAYERS,
    "dropout_rate": DROPOUT_RATE,
    "class_weight_power": EFN_WEIGHT_POWER,
    **{f"precision_{name}": float(val_precision[i]) for i, name in enumerate(class_names)},
    **{f"recall_{name}": float(val_recall[i]) for i, name in enumerate(class_names)},
    "confusion_matrix": str(val_cm.tolist()),
}
write_single_row_csv(VALIDATION_METRICS_CSV, metrics_row)
print(f"Saved validation metrics CSV: {VALIDATION_METRICS_CSV}")

if SAVE_PLOTS:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(val_cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["mild_moderate", "severe"])
    ax.set_yticklabels(["mild_moderate", "severe"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"EPI {FINAL_N_FOLDS}-Fold CV (pooled out-of-fold)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(val_cm[i, j]), ha="center", va="center", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    validation_cm_path = os.path.join(PLOT_DIR, "final_confusion_matrix.png")
    plt.savefig(validation_cm_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved final confusion matrix: {validation_cm_path}")

    roc_path = os.path.join(PLOT_DIR, "roc_precision_recall_severe.png")
    save_roc_and_pr_curves(oof_true_classes, oof_prob_severe, roc_path)
    print(f"Saved pooled OOF severe ROC and precision-recall curves: {roc_path}")

print("\nEPI same-cohort final training complete.")



