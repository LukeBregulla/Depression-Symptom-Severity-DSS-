import os
import gc
import csv
import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, DataCollatorForLanguageModeling, AutoModelForMaskedLM
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from scipy.stats import pearsonr, spearmanr
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


# Data Loading
TRAIN_FILE = "/zi/home/luke.bregulla/Desktop/DSS/data/data_daic_pdch_3class_texts.csv"
TEST_FILE = "/zi/home/luke.bregulla/Desktop/DSS/data/data_epi_en_treatment.csv"

OUTPUT_DIR = "/zi/home/luke.bregulla/Desktop/DSS/pipeline/learning_results_treatment"
LOGGING_DIR = "/zi/home/luke.bregulla/Desktop/DSS/pipeline/learning_results_treatment/logs"
PERFORMANCE_DIR = "/zi/home/luke.bregulla/Desktop/DSS/pipeline/learning_results_treatment/performance_plots"
TREATMENT_GROUP_METRICS_CSV_PATH = os.path.join(OUTPUT_DIR, "epi_treatment_group_metrics.csv")
BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model")

# Load tokenizer
BASE_MODEL = "FacebookAI/xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
DOMAIN_ADAPTED_MODEL_DIR = "/zi/home/luke.bregulla/Desktop/DSS/pipeline/learning_results/domain_adapted_bert"
SAVE_PLOTS = False


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
CLASS_SCORE_ANCHORS = np.asarray([0, 0.5, 1], dtype=np.float32)


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


def permutation_test_group_difference(y_true, y_pred, group_labels, metric_fn, n_permutations=5000, seed=SEED):
    """Two-sided permutation test for metric difference between groups 0 and 1."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    group_labels = np.asarray(group_labels, dtype=np.int64)

    mask0 = group_labels == 0
    mask1 = group_labels == 1
    if mask0.sum() == 0 or mask1.sum() == 0:
        return np.nan, np.nan

    observed = float(metric_fn(y_true[mask0], y_pred[mask0]) - metric_fn(y_true[mask1], y_pred[mask1]))

    rng = np.random.default_rng(seed)
    perm_stats = np.zeros(n_permutations, dtype=float)
    for i in range(n_permutations):
        perm_groups = rng.permutation(group_labels)
        perm_mask0 = perm_groups == 0
        perm_mask1 = perm_groups == 1
        perm_stats[i] = float(metric_fn(y_true[perm_mask0], y_pred[perm_mask0]) - metric_fn(y_true[perm_mask1], y_pred[perm_mask1]))

    p_value = float((np.sum(np.abs(perm_stats) >= abs(observed)) + 1) / (n_permutations + 1))
    return observed, p_value


def write_treatment_group_metrics_csv(output_path, rows):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "n_samples",
                "macro_f1",
                "balanced_accuracy",
                "macro_f1_diff_group0_minus_group1",
                "macro_f1_p_value",
                "balanced_accuracy_diff_group0_minus_group1",
                "balanced_accuracy_p_value",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Dataset loading and splitting
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

train_dataset = load_dataset("csv", data_files={"train": TRAIN_FILE})["train"]
test_dataset = load_dataset("csv", data_files={"test": TEST_FILE}, sep=";", encoding="utf-8")["test"]

train_dataset = train_dataset.select_columns(["text", "labels"])

dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

# Split into train(80%) / validation(20%)
train_indices, val_indices = train_test_split(
    np.arange(len(dataset["train"])),
    test_size=0.2,
    stratify=np.asarray(dataset["train"]["labels"], dtype=np.int64),
    random_state=SEED,
)

dataset = DatasetDict({
    "train": dataset["train"].select(train_indices.tolist()),
    "validation": dataset["train"].select(val_indices.tolist()),
    "test": dataset["test"],
})

print(f"Train rows: {len(dataset['train'])} | Validation rows: {len(dataset['validation'])} | Test rows: {len(dataset['test'])}")

dataset["train"] = dataset["train"].map(lambda x: {"target_class": int(x["labels"])})
dataset["validation"] = dataset["validation"].map(lambda x: {"target_class": int(x["labels"])})

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
        self.register_buffer("class_score_anchors", torch.tensor(CLASS_SCORE_ANCHORS, dtype=torch.float32))

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
            subject_probs = torch.softmax(subject_logits, dim=-1)
            subject_scores = torch.sum(subject_probs * self.class_score_anchors.unsqueeze(0), dim=-1)

            if labels is not None:
                subject_labels = torch.stack(subject_labels_list).long()
                loss = self._classification_loss(subject_logits, subject_labels)
                return {
                    "loss": loss,
                    "logits": subject_logits,
                    "depression_score": subject_scores,
                }

            return {
                "logits": subject_logits,
                "depression_score": subject_scores,
            }

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = self.dropout(outputs.last_hidden_state[:, 0, :])
        chunk_logits = self.classifier(cls_embedding)
        chunk_probs = torch.softmax(chunk_logits, dim=-1)
        chunk_scores = torch.sum(chunk_probs * self.class_score_anchors.unsqueeze(0), dim=-1)
        if labels is not None:
            labels = labels.long()
            loss = self._classification_loss(chunk_logits, labels)
            return {
                "loss": loss,
                "logits": chunk_logits,
                "depression_score": chunk_scores,
            }
        return {
            "logits": chunk_logits,
            "depression_score": chunk_scores,
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

def extract_pred_scores(predictions):
    """Trainer predictions can be an array or tuple; take logits/scores consistently."""
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]
    return logits_to_scores(predictions)


def extract_pred_classes(predictions):
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]
    logits = np.asarray(predictions, dtype=float)
    if logits.ndim == 1:
        logits = logits.reshape(-1, NUM_CLASSES)
    return np.argmax(logits, axis=1).astype(np.int64)


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


def _latest_checkpoint_dir(output_dir):
    if not os.path.isdir(output_dir):
        return None
    candidates = []
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if not (name.startswith("checkpoint-") and os.path.isdir(path)):
            continue
        try:
            step = int(name.split("-")[-1])
        except ValueError:
            continue
        if os.path.isfile(os.path.join(path, "model.safetensors")) or os.path.isfile(os.path.join(path, "pytorch_model.bin")):
            candidates.append((step, path))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


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
    logging_dir=LOGGING_DIR,
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

print(
    "Effective BERT params: "
    f"lr={training_args.learning_rate}, warmup_ratio={training_args.warmup_ratio}, "
    f"epochs={training_args.num_train_epochs}, weight_decay={training_args.weight_decay}, "
    f"batch={training_args.per_device_train_batch_size}",
    flush=True,
)
print(
    "Effective MLM params: "
    f"lr={mlm_args.learning_rate}, warmup_steps={mlm_args.warmup_steps}, "
    f"epochs={mlm_args.num_train_epochs}, weight_decay={mlm_args.weight_decay}, "
    f"batch={mlm_args.per_device_train_batch_size}",
    flush=True,
)

domain_model_ready = _model_dir_has_weights(DOMAIN_ADAPTED_MODEL_DIR)
if domain_model_ready:
    print(f"Skipping MLM warmup: found existing domain-adapted model at {DOMAIN_ADAPTED_MODEL_DIR}")
else:
    # Use DAIC training text only for MLM warmup (no EPI text usage here).
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
val_true_classes = np.asarray(dataset["validation"]["target_class"], dtype=np.int64)

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
if SAVE_PLOTS:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(val_cm, cmap="Blues")
    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["none", "moderate", "severe"])
    ax.set_yticklabels(["none", "moderate", "severe"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
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

os.makedirs(PERFORMANCE_DIR, exist_ok=True)
if did_train_classifier and best_ckpt is not None:
    import shutil
    output_dir = training_args.output_dir
    for name in os.listdir(output_dir):
        ckpt_dir = os.path.join(output_dir, name)
        if name.startswith("checkpoint-") and os.path.isdir(ckpt_dir) and ckpt_dir != best_ckpt:
            shutil.rmtree(ckpt_dir)
            print(f"Deleted: {ckpt_dir}")


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Evaluation on EPI treatment data (grouped metrics only)
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Predict on treatment test set (subject-level).
predictions = trainer.predict(dataset["test"])
test_pred_classes = extract_pred_classes(predictions.predictions)

required_columns = {"hamd_sum", "treatment_group"}
if required_columns.issubset(set(dataset["test"].column_names)):
    hamd_true = np.asarray(dataset["test"]["hamd_sum"], dtype=float)
    hamd_true_classes = np.where(hamd_true <= 7, 0, np.where(hamd_true <= 23, 1, 2)).astype(np.int64)
    treatment_group = np.asarray(dataset["test"]["treatment_group"], dtype=np.int64)

    rows = []
    for group_id in [0, 1]:
        group_mask = treatment_group == group_id
        y_true_g = hamd_true_classes[group_mask]
        y_pred_g = test_pred_classes[group_mask]

        if y_true_g.size == 0:
            group_macro_f1 = np.nan
            group_bal_acc = np.nan
        else:
            group_macro_f1 = float(f1_score(y_true_g, y_pred_g, labels=[0, 1, 2], average="macro", zero_division=0))
            group_bal_acc = balanced_accuracy(y_true_g, y_pred_g)

        rows.append(
            {
                "group": int(group_id),
                "n_samples": int(y_true_g.size),
                "macro_f1": group_macro_f1,
                "balanced_accuracy": group_bal_acc,
                "macro_f1_diff_group0_minus_group1": np.nan,
                "macro_f1_p_value": np.nan,
                "balanced_accuracy_diff_group0_minus_group1": np.nan,
                "balanced_accuracy_p_value": np.nan,
            }
        )

    f1_diff, f1_p = permutation_test_group_difference(
        hamd_true_classes,
        test_pred_classes,
        treatment_group,
        lambda y_t, y_p: float(f1_score(y_t, y_p, labels=[0, 1, 2], average="macro", zero_division=0)),
    )
    bal_diff, bal_p = permutation_test_group_difference(
        hamd_true_classes,
        test_pred_classes,
        treatment_group,
        lambda y_t, y_p: balanced_accuracy(y_t, y_p),
    )

    for row in rows:
        row["macro_f1_diff_group0_minus_group1"] = f1_diff
        row["macro_f1_p_value"] = f1_p
        row["balanced_accuracy_diff_group0_minus_group1"] = bal_diff
        row["balanced_accuracy_p_value"] = bal_p

    write_treatment_group_metrics_csv(TREATMENT_GROUP_METRICS_CSV_PATH, rows)

    print("\n" + "="*60)
    print("TEST EVALUATION (EPI TREATMENT GROUPS)")
    print("="*60)
    for row in rows:
        print(f"Group {row['group']} | n={row['n_samples']} | Macro F1={row['macro_f1']:.4f} | Balanced Accuracy={row['balanced_accuracy']:.4f}")
    print(f"Macro F1 difference (group 0 - group 1): {f1_diff:.4f} | permutation p-value: {f1_p:.4g}")
    print(f"Balanced Accuracy difference (group 0 - group 1): {bal_diff:.4f} | permutation p-value: {bal_p:.4g}")
    print(f"Saved treatment-group metrics CSV: {TREATMENT_GROUP_METRICS_CSV_PATH}")
else:
    print("\nRequired columns missing in test set. Expected: hamd_sum, treatment_group")



