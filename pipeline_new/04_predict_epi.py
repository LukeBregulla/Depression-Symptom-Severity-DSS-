"""
04_predict_epi.py - Standalone EPI prediction for trained models (02 or 03)
Loads a pre-trained model and evaluates it on EPI without any training.

Usage:
  python 04_predict_epi.py                                    # Uses defaults
  python 04_predict_epi.py --model_dir /path/to/model         # Custom model
  python 04_predict_epi.py --model_dir /path/to/model --output_dir /path/to/output
"""

import os
import gc
import csv
import re
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModel
from datasets import load_dataset
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score, average_precision_score
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput


# ============================================================================
# CONFIG - ADJUST THESE FOR YOUR NEEDS
# ============================================================================

# Default model to load (change to 02 or 03 best_model)
DEFAULT_MODEL_DIR = "/zi/home/luke.bregulla/Desktop/DSS/results_new/03_pdch/best_model"

# Default output directory for results
DEFAULT_OUTPUT_DIR = "/zi/home/luke.bregulla/Desktop/DSS/results_new/04_pdch"

# EPI dataset path (should be test split with labels)
EPI_FILE = "/zi/home/luke.bregulla/Desktop/DSS/data_new/data_epi.csv"

# Model and encoding config (must match training pipeline)
BASE_MODEL = "microsoft/deberta-v3-base"
MAX_LEN = 512
STRIDE = 256
SUBJECT_BATCH_SIZE = 8
CHUNK_MICRO_BATCH_SIZE = 8
FREEZE_BOTTOM_LAYERS = 4
USE_GRADIENT_CHECKPOINTING = False
NUM_CLASSES = 2
LABEL_SPACE = [0, 1]
SEED = 42

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Georgia", "DejaVu Serif", "Times New Roman", "serif"]


# ============================================================================
# Model Definition (matches training pipeline)
# ============================================================================

class CustomBERTModel(nn.Module):
    """Chunk-level encoding, subject mean pooling, and binary classification head."""

    def __init__(
        self,
        pretrained_model_name,
        class_weights,
        chunk_micro_batch_size=CHUNK_MICRO_BATCH_SIZE,
        freeze_bottom_layers=FREEZE_BOTTOM_LAYERS,
        dropout_rate=0.3,
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

            if labels is not None:
                subject_labels = torch.stack(subject_labels_list).long()
                loss = torch.nn.functional.cross_entropy(
                    subject_logits,
                    subject_labels.long(),
                    weight=self.class_weights,
                )
                return SequenceClassifierOutput(loss=loss, logits=subject_logits)

            return SequenceClassifierOutput(logits=subject_logits)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = self.dropout(outputs.last_hidden_state[:, 0, :])
        cls_embedding = cls_embedding.to(self.classifier.weight.dtype)
        chunk_logits = self.classifier(cls_embedding)
        if labels is not None:
            labels = labels.long()
            loss = torch.nn.functional.cross_entropy(
                chunk_logits,
                labels.long(),
                weight=self.class_weights,
            )
            return SequenceClassifierOutput(loss=loss, logits=chunk_logits)
        return SequenceClassifierOutput(logits=chunk_logits)


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


# ============================================================================
# Utility Functions
# ============================================================================

def participant_key(sample_id):
    """Collapse session-level ids (ZI002-1, PDCH_001A) to one key per participant."""
    return re.sub(r"(-\d+|(?<=\d)[A-Za-z])$", "", str(sample_id))


def permutation_participant_wise(y_true, y_pred, participant_ids, n_permutations=10000, seed=42):
    """Permutation test for macro F1 that preserves whole participant session blocks."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    participant_ids = np.asarray([participant_key(sample_id) for sample_id in participant_ids])
    observed = float(f1_score(y_true, y_pred, labels=LABEL_SPACE, average="macro", zero_division=0))

    blocks = [np.where(participant_ids == participant)[0] for participant in np.unique(participant_ids)]
    size_groups = {}
    for block in blocks:
        size_groups.setdefault(block.size, []).append(block)

    rng = np.random.default_rng(seed)
    perm_scores = np.empty(n_permutations, dtype=np.float64)
    for permutation_index in range(n_permutations):
        permuted_true = np.empty_like(y_true)
        for group in size_groups.values():
            order = rng.permutation(len(group))
            for target_index, source_index in enumerate(order):
                permuted_true[group[target_index]] = y_true[group[source_index]]
        perm_scores[permutation_index] = f1_score(permuted_true, y_pred, labels=LABEL_SPACE, average="macro", zero_division=0)

    return {
        "observed_macro_f1": observed,
        "null_macro_f1_mean": float(np.mean(perm_scores)),
        "perm_p_value": float((np.sum(perm_scores >= observed) + 1) / (n_permutations + 1)),
        "n_permutations": int(n_permutations),
    }


def safe_float(val):
    """Safely convert to float, return NaN if fails."""
    try:
        f = float(val)
        return f if np.isfinite(f) else np.nan
    except (ValueError, TypeError):
        return np.nan


def load_csv_dataset(path, split):
    """Load CSV with auto-detected delimiter."""
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
    sep = "\t" if header.count("\t") > header.count(",") else ","
    return load_dataset("csv", data_files={split: path}, sep=sep, encoding="utf-8")[split]


def extract_pred_and_prob(predictions):
    """Extract class predictions and probability matrix."""
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]
    logits = np.asarray(predictions, dtype=float)
    if logits.ndim == 1:
        logits = logits.reshape(-1, NUM_CLASSES)
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    pred_classes = np.argmax(logits, axis=1).astype(np.int64)
    return pred_classes, probs


# ============================================================================
# Main Evaluation Function
# ============================================================================

def evaluate_on_epi(model_dir: str, output_dir: str) -> dict:
    """
    Load a trained model and evaluate on EPI dataset.
    
    Args:
        model_dir: Path to directory containing model weights and tokenizer
        output_dir: Path to save evaluation results and plots
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*70}")
    print(f"EPI EVALUATION")
    print(f"{'='*70}")
    print(f"Model directory:  {model_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Dataset:          {EPI_FILE}")
    
    # Verify model directory exists
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    os.makedirs(output_dir, exist_ok=True)

    # ========================================================================
    # Load EPI dataset
    # ========================================================================
    print(f"\nLoading EPI dataset...")
    epi_ds = load_csv_dataset(EPI_FILE, "test")
    
    if "labels" not in epi_ds.column_names:
        raise ValueError("EPI file must contain 'labels' column for evaluation")

    labels = np.asarray(epi_ds["labels"], dtype=np.int64).reshape(-1)
    if sorted(np.unique(labels).tolist()) != LABEL_SPACE:
        raise ValueError(f"EPI labels must be {LABEL_SPACE}, got {sorted(np.unique(labels).tolist())}")

    n_epi = len(labels)
    print(f"  Loaded {n_epi} EPI samples")
    print(f"  Class distribution: {np.bincount(labels, minlength=NUM_CLASSES).tolist()}")

    # Prepare dataframe
    epi_df = pd.DataFrame({
        "id": [str(x) for x in epi_ds["id"]],
        "text": [str(x) for x in epi_ds["text"]],
        "labels": labels.astype(int),
    })

    if "continuous_score" not in epi_ds.column_names:
        raise ValueError("EPI dataset missing 'continuous_score' column")
    epi_df["continuous_score"] = [safe_float(x) for x in epi_ds["continuous_score"]]

    # Save temp CSV and reload for consistency
    tmp_epi_csv = os.path.join(output_dir, "_tmp_epi_test.csv")
    epi_df.to_csv(tmp_epi_csv, index=False)
    test_ds = load_csv_dataset(tmp_epi_csv, "test")
    test_ds = test_ds.map(
        lambda x: {
            "id": str(x["id"]),
            "text": str(x["text"]),
            "target_class": int(x["labels"]),
            "continuous_score": safe_float(x["continuous_score"]),
        },
        remove_columns=test_ds.column_names,
    )

    # ========================================================================
    # Load model
    # ========================================================================
    print(f"\nLoading model from: {model_dir}")
    model = CustomBERTModel(
        BASE_MODEL,
        class_weights=np.ones(NUM_CLASSES, dtype=np.float32),
        chunk_micro_batch_size=CHUNK_MICRO_BATCH_SIZE,
        freeze_bottom_layers=FREEZE_BOTTOM_LAYERS,
        dropout_rate=0.3,
        use_gradient_checkpointing=False,
    )

    # Try safetensors first, then pytorch bin
    safe_path = os.path.join(model_dir, "model.safetensors")
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    
    if os.path.isfile(safe_path):
        print(f"  Loading from: {safe_path}")
        from safetensors.torch import load_file as load_safetensors
        state_dict = load_safetensors(safe_path)
    elif os.path.isfile(bin_path):
        print(f"  Loading from: {bin_path}")
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise RuntimeError(f"No model weights found in {model_dir}. Looked for: {safe_path}, {bin_path}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing checkpoint weights: {missing[:5]}")
    if unexpected:
        print(f"  Note: ignored {len(unexpected)} unexpected keys during load")

    # ========================================================================
    # Run inference
    # ========================================================================
    print(f"\nRunning inference on EPI...")
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=SUBJECT_BATCH_SIZE,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=SubjectChunkingCollator(tokenizer, max_len=MAX_LEN, stride=STRIDE),
    )

    test_out = trainer.predict(test_ds)
    test_pred, test_prob = extract_pred_and_prob(test_out.predictions)
    test_true = np.asarray(test_ds["target_class"], dtype=np.int64)
    
    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========================================================================
    # Compute metrics
    # ========================================================================
    print(f"\nComputing metrics...")
    
    macro_f1 = float(f1_score(test_true, test_pred, labels=LABEL_SPACE, average="macro", zero_division=0))
    cm = confusion_matrix(test_true, test_pred, labels=LABEL_SPACE)
    balanced_acc_per_class = cm.diagonal() / cm.sum(axis=1)
    balanced_acc = float(np.mean(balanced_acc_per_class))
    precision = precision_score(test_true, test_pred, labels=LABEL_SPACE, average=None, zero_division=0)
    recall = recall_score(test_true, test_pred, labels=LABEL_SPACE, average=None, zero_division=0)
    
    try:
        roc_auc = float(roc_auc_score(test_true, test_prob[:, 1]))
        average_precision = float(average_precision_score(test_true, test_prob[:, 1]))
    except Exception as e:
        print(f"  Warning: ROC-AUC or average-precision computation failed: {e}")
        roc_auc = np.nan
        average_precision = np.nan

    # Continuous scores are exported per subject; the association analysis is done separately.
    cont_scores = np.asarray(test_ds["continuous_score"], dtype=np.float64)
    perm_result = permutation_participant_wise(
        test_true,
        test_pred,
        participant_ids=np.asarray(test_ds["id"], dtype=object),
        n_permutations=10000,
        seed=SEED,
    )

    # ========================================================================
    # Save results
    # ========================================================================
    
    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["mild_moderate", "severe"])
    ax.set_yticklabels(["mild_moderate", "severe"])
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("EPI Test Confusion Matrix", fontsize=12)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", 
                   color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=180, bbox_inches="tight")
    plt.close()

    # Severe-positive ROC and precision-recall curves.
    try:
        from sklearn.metrics import precision_recall_curve, roc_curve
        fpr, tpr, _ = roc_curve(test_true, test_prob[:, 1], pos_label=1)
        pr_precision, pr_recall, _ = precision_recall_curve(test_true, test_prob[:, 1], pos_label=1)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        axes[0].plot(fpr, tpr, color="#1F77B4", lw=2, label=f"AUC = {roc_auc:.3f}")
        axes[0].plot([0, 1], [0, 1], color="gray", lw=1.2, linestyle="--")
        axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC: Severe vs Mild/Moderate")
        axes[0].legend(loc="lower right", frameon=False)
        axes[0].grid(alpha=0.2)
        axes[1].plot(pr_recall, pr_precision, color="#FF7F0E", lw=2, label=f"Average precision = {average_precision:.3f}")
        axes[1].axhline(np.mean(test_true == 1), color="gray", lw=1.2, linestyle="--", label="Severe prevalence")
        axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision", title="Precision-Recall: Severe vs Mild/Moderate")
        axes[1].legend(loc="best", frameon=False)
        axes[1].grid(alpha=0.2)
        fig.suptitle("External EPI Evaluation", y=1.03)
        fig.tight_layout()
        roc_path = os.path.join(output_dir, "roc_precision_recall_severe.png")
        fig.savefig(roc_path, dpi=180, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not generate severe ROC/precision-recall curves: {e}")
        roc_path = None

    # Class distribution histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = np.bincount(test_true, minlength=2)
    bars = ax.bar(["Mild/Moderate", "Severe"], counts, color=["#1f77b4", "#ff7f0e"], alpha=0.7, edgecolor="black")
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("EPI Test Set Class Distribution", fontsize=12)
    ax.set_ylim([0, max(counts) * 1.15])
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    dist_path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(dist_path, dpi=180, bbox_inches="tight")
    plt.close()

    # Probability distribution plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Probability for mild/moderate class
    axes[0].hist(test_prob[test_true == 0, 0], bins=15, alpha=0.6, label="True mild/moderate", color="blue", edgecolor="black")
    axes[0].hist(test_prob[test_true == 1, 0], bins=15, alpha=0.6, label="True severe", color="red", edgecolor="black")
    axes[0].set_xlabel("Predicted probability", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].set_title("Mild/Moderate Class Probability Distribution", fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis="y")
    
    # Probability for severe class
    axes[1].hist(test_prob[test_true == 0, 1], bins=15, alpha=0.6, label="True mild/moderate", color="blue", edgecolor="black")
    axes[1].hist(test_prob[test_true == 1, 1], bins=15, alpha=0.6, label="True severe", color="red", edgecolor="black")
    axes[1].set_xlabel("Predicted probability", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].set_title("Severe Class Probability Distribution", fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    prob_path = os.path.join(output_dir, "probability_distributions.png")
    plt.savefig(prob_path, dpi=180, bbox_inches="tight")
    plt.close()

    # Metrics CSV
    metrics_dict = {
        "n_samples": int(n_epi),
        "n_mild_moderate": int(np.sum(test_true == 0)),
        "n_severe": int(np.sum(test_true == 1)),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_acc),
        "precision_mild_moderate": float(precision[0]),
        "precision_severe": float(precision[1]),
        "recall_mild_moderate": float(recall[0]),
        "recall_severe": float(recall[1]),
        "roc_auc_severe": float(roc_auc),
        "average_precision_severe": float(average_precision),
        "perm_observed_macro_f1": perm_result["observed_macro_f1"],
        "perm_null_macro_f1_mean": perm_result["null_macro_f1_mean"],
        "perm_p_value": perm_result["perm_p_value"],
        "perm_n_permutations": perm_result["n_permutations"],
    }
    
    metrics_path = os.path.join(output_dir, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_dict.keys()))
        writer.writeheader()
        writer.writerow(metrics_dict)

    severe_metrics_path = os.path.join(output_dir, "severe_discrimination_metrics.csv")
    with open(severe_metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["positive_class", "support", "precision", "recall", "roc_auc", "average_precision"])
        writer.writeheader()
        writer.writerow({
            "positive_class": "severe",
            "support": int(np.sum(test_true == 1)),
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "roc_auc": float(roc_auc),
            "average_precision": float(average_precision),
        })

    # Predictions CSV
    pred_path = os.path.join(output_dir, "subject_predictions.csv")
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "participant", "true_label", "pred_label",
            "p_mild_moderate", "p_severe", "continuous_score",
        ])
        writer.writeheader()
        for i in range(len(test_ds)):
            writer.writerow({
                "id": test_ds["id"][i],
                "participant": participant_key(test_ds["id"][i]),
                "true_label": int(test_true[i]),
                "pred_label": int(test_pred[i]),
                "p_mild_moderate": float(test_prob[i, 0]),
                "p_severe": float(test_prob[i, 1]),
                "continuous_score": float(cont_scores[i]),
            })

    # ========================================================================
    # Print results
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Macro F1:              {macro_f1:.4f}")
    print(f"Balanced Accuracy:     {balanced_acc:.4f}")
    print(f"ROC-AUC (severe):      {roc_auc:.4f}")
    print(f"Average precision:     {average_precision:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nPer-class Metrics:")
    print(f"  mild_moderate: precision={precision[0]:.4f}, recall={recall[0]:.4f}")
    print(f"  severe:        precision={precision[1]:.4f}, recall={recall[1]:.4f}")
    print(f"\nOutput Files:")
    print(f"  Metrics:                  {metrics_path}")
    print(f"  Predictions:              {pred_path}")
    print(f"  Confusion Matrix:         {cm_path}")
    print(f"  ROC AUC Curve:            {roc_path if roc_path else '(not generated)'}")
    print(f"  Class Distribution:       {dist_path}")
    print(f"  Probability Distributions: {prob_path}")

    return metrics_dict


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on EPI dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 04_predict_epi.py                                    # Uses defaults
  python 04_predict_epi.py --model_dir /path/to/model         # Custom model
  python 04_predict_epi.py --model_dir /path/to/02_model --output_dir /path/to/02_results
        """
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=f"Path to saved model directory (default: {DEFAULT_MODEL_DIR})"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Path to save evaluation results (default: {DEFAULT_OUTPUT_DIR})"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

    evaluate_on_epi(args.model_dir, args.output_dir)
    print(f"\n✓ Evaluation complete. Results saved to: {args.output_dir}\n")
