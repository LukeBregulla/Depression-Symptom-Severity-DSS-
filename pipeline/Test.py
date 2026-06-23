import os
import csv
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, roc_curve, auc, precision_score, recall_score
from safetensors.torch import load_file as load_safetensors

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Georgia", "DejaVu Serif", "Times New Roman", "serif"]

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Config and Setup
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#load config
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

cfg = load_config()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_FILE = str(PROJECT_ROOT / cfg["test_data"]["file"])
TEST_SEP = cfg["test_data"]["sep"]
TEST_ENCODING = cfg["test_data"]["encoding"]
OUTPUT_DIR = str(PROJECT_ROOT / cfg["output"]["directory"])
OUTPUT_NAME = cfg["output"]["name"]
NUM_CLASSES = cfg["evaluation"]["classes"]

# trained model dir
BASE_MODEL = "FacebookAI/xlm-roberta-base"
DOMAIN_ADAPTED_BERT_DIR = str(PROJECT_ROOT / "results/Validation/domain_adapted_bert")
BEST_MODEL_DIR = str(PROJECT_ROOT / "results/Validation/best_model")

# Model always outputs 3 classes (was trained on 3 classes); config sets classes for metric calculation
MODEL_NUM_CLASSES = 3

# Map evaluation classes to label indices
METRIC_LABELS = {1: [0], 2: [1, 2], 3: [0, 1, 2]}.get(NUM_CLASSES)
if METRIC_LABELS is None:
    raise ValueError(f"Invalid classes count: {NUM_CLASSES}. Must be 1, 2, or 3.")


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Utility Functions
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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


#precision recall auc roc
def precision_recall_auc_roc(output_path, true_classes, pred_probs, metric_labels=None):
    metric_labels = metric_labels or [0, 1, 2]
    class_names = ["none", "moderate", "severe"]
    pred_classes = np.argmax(pred_probs, axis=1)
    
    rows = []
    for class_id in metric_labels:
        y_bin = (true_classes == class_id).astype(int)
        y_pred_bin = (pred_classes == class_id).astype(int)
        y_score = pred_probs[:, class_id]
        precision = precision_score(y_bin, y_pred_bin, zero_division=0)
        recall = recall_score(y_bin, y_pred_bin, zero_division=0)
        roc_auc = roc_auc_score(y_bin, y_score) if len(np.unique(y_bin)) > 1 else float('nan')
        rows.append({"class_id": class_id, "class_name": class_names[class_id], "precision": precision, "recall": recall, "roc_auc": roc_auc})
    
    write_csv(output_path, ["class_id", "class_name", "precision", "recall", "roc_auc"], rows)


# plot auc roc
def save_roc_curves(y_true, y_scores, output_path, macro_auc, metric_labels=None):
    metric_labels = metric_labels or [0, 1, 2]
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_scores = np.asarray(y_scores, dtype=float)
    if y_scores.ndim != 2 or y_scores.shape[1] != 3:
        raise ValueError("y_scores must have shape (n_samples, 3)")

    n_classes = len(metric_labels)
    fig_width = 5 * n_classes if n_classes > 1 else 5
    fig, axes = plt.subplots(1, n_classes, figsize=(fig_width, 4.6), sharex=True, sharey=True)
    axes = [axes] if n_classes == 1 else axes 
    
    for idx, class_id in enumerate(metric_labels):
        ax = axes[idx]
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
            continue

        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        class_auc = auc(fpr, tpr)

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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

# permutation test for macro F1 vs chance
def macro_f1_present_labels(y_true, y_pred, label_space=None):
    """Compute macro F1 on the labels present in y_true or specified in label_space."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size == 0:
        return 0.0
    labels = np.asarray(np.unique(y_true), dtype=np.int64) if label_space is None else np.asarray(label_space, dtype=np.int64)
    return float(f1_score(y_true, y_pred, labels=labels.tolist(), average="macro", zero_division=0))

def permutation_subject_wise(y_true, y_pred, subject_ids=None, n_permutations=10000, seed=42, label_space=None):
    """Permutation test for macro-F1 vs chance on subjects."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if subject_ids is None:
        subject_ids = np.arange(y_true.size, dtype=np.int64)
    subject_ids = np.asarray(subject_ids)
    if subject_ids.size != y_true.size:
        raise ValueError("subject_ids length must match y_true length")

    if label_space is None:
        label_space = np.unique(y_true)
    else:
        label_space = np.asarray(label_space, dtype=np.int64)
    
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
    null_mean = float(np.mean(perm_scores))

    p_value_right = float((np.sum(perm_scores >= observed) + 1) / (n_permutations + 1))
    return {
        "n_samples": int(y_true.size),
        "observed_macro_f1": observed,
        "null_macro_f1_mean": null_mean,
        "p": p_value_right,
        "n_permutations": int(n_permutations),
        "n_subjects": int(unique_subjects.size),
        "permutation_unit": "subject",
        "label_space": "|".join([str(int(x)) for x in label_space.tolist()]),
        "note": "Permutation test (subject-wise): subject labels are permuted against fixed predictions and broadcast to all sessions of each subject.",
    }


# plot class distribution
def save_class_distribution_plot(distribution_map, output_path):
    labels = ["none", "moderate", "severe"]
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
test_dataset = load_dataset("csv", data_files={"test": TEST_FILE}, sep=TEST_SEP, encoding=TEST_ENCODING)["test"]
print(f"Test set size: {len(test_dataset)}")

# Extract HAMD scores and convert to 3 class
if "hamd_sum" not in test_dataset.column_names:
    raise ValueError("hamd_sum column not found in test data")
hamd_scores = np.asarray(test_dataset["hamd_sum"], dtype=float)
test_true_classes = np.where(hamd_scores <= 7, 0, np.where(hamd_scores <= 23, 1, 2)).astype(np.int64)

# Filter to evaluation labels if not using all 3 classes
if NUM_CLASSES < MODEL_NUM_CLASSES:
    eval_labels = np.array(METRIC_LABELS, dtype=np.int64)
    mask = np.isin(test_true_classes, eval_labels)
    test_dataset = test_dataset.select(np.where(mask)[0].tolist())
    test_true_classes = test_true_classes[mask]
    print(f"After filtering to evaluation_labels {tuple(eval_labels)}: {len(test_dataset)} samples remain")

# Use row indices as IDs
test_row_ids = [str(i) for i in range(len(test_dataset))]

# Keep only text column for inference
test_dataset = test_dataset.select_columns(["text"])
print(f"Final test set size: {len(test_dataset)}")


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Model Classes (from Train.py)
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class CustomBERTModel(nn.Module):

    def __init__(self, pretrained_model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.config = self.encoder.config
        self.config.num_labels = MODEL_NUM_CLASSES
        self.chunk_micro_batch_size = 16
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.config.hidden_size, MODEL_NUM_CLASSES)

    @classmethod
    def from_checkpoint(cls, pretrained_model_name, checkpoint_dir):
        model = cls(pretrained_model_name)
        safe_path = os.path.join(checkpoint_dir, "model.safetensors")
        bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if os.path.isfile(safe_path):
            state_dict = load_safetensors(safe_path)
        elif os.path.isfile(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise RuntimeError(f"Failed to load checkpoint from {checkpoint_dir}")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        for key_type, keys in [("missing", missing), ("unexpected", unexpected)]:
            if keys:
                print(f"Checkpoint load warning: {key_type} keys={len(keys)}")
        return model

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
                subject_labels = torch.stack(subject_labels_list).long()
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
        return {
            "logits": chunk_logits,
        }

# chunking and tracking for subject level aggregation
class SubjectChunkingCollator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_len = 512
        self.stride = 256

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
# Load Model
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print(f"\nLoading model from: {BEST_MODEL_DIR}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = CustomBERTModel.from_checkpoint(DOMAIN_ADAPTED_BERT_DIR, BEST_MODEL_DIR)
model.eval()
print(f"✓ Model loaded successfully")


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Inference
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_eval_batch_size=16,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=SubjectChunkingCollator(tokenizer),
)

print(f"\nRunning inference on {len(test_dataset)} samples...")
predictions = trainer.predict(test_dataset)

# Extract pred classes and probs from logits
logits = np.asarray(predictions.predictions[0] if isinstance(predictions.predictions, (tuple, list)) else predictions.predictions, dtype=float)
if logits.ndim == 1:
    logits = logits.reshape(-1, MODEL_NUM_CLASSES)
test_pred_classes = np.argmax(logits, axis=1).astype(np.int64)
logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
test_pred_probs = np.exp(logits_shifted) / np.sum(np.exp(logits_shifted), axis=1, keepdims=True)

print(f"Predictions complete")


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Metrics and Output Files
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Compute confusion matrix and class metrics
test_cm = confusion_matrix(test_true_classes, test_pred_classes, labels=METRIC_LABELS).astype(float)
recalls = [(test_cm[i, i] / test_cm[i, :].sum()) if test_cm[i, :].sum() > 0 else 0.0 for i in range(len(METRIC_LABELS))]
test_bal_acc = np.mean(recalls) if recalls else 0.0
test_macro_f1 = f1_score(test_true_classes, test_pred_classes, labels=METRIC_LABELS, average="macro", zero_division=0)
try:
    roc_auc = roc_auc_score(test_true_classes, test_pred_probs, multi_class="ovr", average="macro", zero_division=0)
except:
    roc_auc = float("nan")

print("\n" + "="*60)
print("TEST RESULTS")
print("="*60)
print(f"Balanced Accuracy: {test_bal_acc:.4f}")
print(f"Macro F1:          {test_macro_f1:.4f}")
print("Confusion Matrix (rows=true, cols=pred):")
print(test_cm)

# Permutation test (subject-wise to account for autocorrelation)
perm_result = permutation_subject_wise(test_true_classes, test_pred_classes, subject_ids=np.asarray(test_row_ids), n_permutations=10000, seed=42, label_space=METRIC_LABELS)
test_p_value = perm_result["p"]
test_null_mean = perm_result["null_macro_f1_mean"]

p_value_str = f"{test_p_value:.4f}"

print(f"Permutation test (subject-wise):")
print(f"  n_permutations: {perm_result['n_permutations']}, Null distribution mean: {test_null_mean:.4f}, p-value: {p_value_str}")

# Save all output files
os.makedirs(OUTPUT_DIR, exist_ok=True)

# individual subject predictions
subject_pred_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}_subject_predictions.csv")
rows = [
    {
        "index": str(row_id),
        "true_class": int(y_t),
        "pred_class": int(y_p),
        "p_none": float(test_pred_probs[i, 0]),
        "p_moderate": float(test_pred_probs[i, 1]),
        "p_severe": float(test_pred_probs[i, 2]),
    }
    for i, (row_id, y_t, y_p) in enumerate(zip(test_row_ids, test_true_classes, test_pred_classes))
]
write_csv(subject_pred_path, ["index", "true_class", "pred_class", "p_none", "p_moderate", "p_severe"], rows)
print(f"✓ Saved: {subject_pred_path}")

# overall metrics
overall_metrics_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}_overall_metrics.csv")
write_csv(
    overall_metrics_path,
    ["n", "balanced_accuracy", "macro_f1", "roc_auc", "p", "null_macro_f1_mean"],
    {
        "n": len(test_true_classes),
        "balanced_accuracy": test_bal_acc,
        "macro_f1": test_macro_f1,
        "roc_auc": roc_auc,
        "p": test_p_value,
        "null_macro_f1_mean": test_null_mean,
    },
)
print(f"✓ Saved: {overall_metrics_path}")

# Precision, recall, ROC-AUC CSV
prec_rec_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}_precision_recall_roc_auc.csv")
precision_recall_auc_roc(
    prec_rec_path,
    test_true_classes,
    test_pred_probs,
    metric_labels=METRIC_LABELS,
)
print(f"✓ Saved: {prec_rec_path}")

# CM
perf_dir = os.path.join(OUTPUT_DIR, "performance_plots")
os.makedirs(perf_dir, exist_ok=True)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(test_cm, cmap="Blues")
class_labels = [["none", "moderate", "severe"][i] for i in METRIC_LABELS]
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

# plot AUC ROC
roc_path = os.path.join(perf_dir, "roc_curves.png")
save_roc_curves(
    test_true_classes,
    test_pred_probs,
    roc_path,
    roc_auc,
    metric_labels=METRIC_LABELS,
)
print(f"✓ Saved: {roc_path}")

# plot Class distribution
class_dist = {"test": np.bincount(np.asarray(test_true_classes, dtype=np.int64), minlength=3)}
class_dist_path = os.path.join(perf_dir, "class_distribution.png")
save_class_distribution_plot(class_dist, class_dist_path)
print(f"✓ Saved: {class_dist_path}")


print("\n" + "="*60)
print("INFERENCE COMPLETE")
print("="*60)
