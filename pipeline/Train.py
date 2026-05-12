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


# Data Loading
TRAIN_FILE = "/zi/home/luke.bregulla/Desktop/data/data_daic_pdch_3class_texts.csv"
TEST_FILE = "/zi/home/luke.bregulla/Desktop/data/data_epi_en_clean.csv"

OUTPUT_DIR = "/zi/home/luke.bregulla/Desktop/predict_hamd/learning_results"
LOGGING_DIR = "/zi/home/luke.bregulla/Desktop/predict_hamd/learning_results/logs"
PERFORMANCE_DIR = "/zi/home/luke.bregulla/Desktop/predict_hamd/learning_results/performance_plots"
OVERALL_METRICS_CSV_PATH = os.path.join(OUTPUT_DIR, "epi_holdout_overall_metrics.csv")
PER_SESSION_METRICS_CSV_PATH = os.path.join(OUTPUT_DIR, "epi_holdout_metrics_by_session.csv")
BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model")

# Load tokenizer
BASE_MODEL = "FacebookAI/xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
DOMAIN_ADAPTED_MODEL_DIR = "/zi/home/luke.bregulla/Desktop/predict_hamd/learning_results/domain_adapted_bert"
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
# Evaluation on Holdout EPI data
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Predict on test set (subject-level, no pre-chunking needed)
predictions = trainer.predict(dataset["test"])

test_scores = extract_pred_scores(predictions.predictions)
test_pred_classes = extract_pred_classes(predictions.predictions)

# EPI test: convert HAM-D to 3 classes (0-7: none, 8-23: moderate, 24+: severe)
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

    print("\n" + "="*60)
    print("TEST EVALUATION (EPI HOLDOUT)")
    print("="*60)
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
        ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(["none", "moderate", "severe"])
        ax.set_yticklabels(["none", "moderate", "severe"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
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

else:
    print("\nHAM-D column not found in test set; skipped test evaluation.")


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
    """One scatter subplot per session group."""
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

        # Pearson is undefined if one side is constant; keep the plot and report neutral stats.
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


if "hamd_sum" in dataset["test"].column_names:
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
            session_plot_path = os.path.join(PERFORMANCE_DIR, "epi_hamd_scatter_by_session.png")
            save_session_grouped_epi_scatter(
                x_values=np.asarray(test_scores, dtype=float),
                y_values=np.asarray(hamd_true, dtype=float),
                groups=session_labels,
                output_path=session_plot_path,
                title=f"EPI: Predicted continuous score (0-1) vs True HAM-D by session  (rho={rho_hamd:.3f}, p={p_hamd:.3g})",
                x_label="Predicted continuous score (0-1)",
            )
            print(f"Saved EPI session scatter plot: {session_plot_path}")

            session_grid_path = os.path.join(PERFORMANCE_DIR, "epi_hamd_cloud_grid_by_session.png")
            n_session_panels = save_session_cloud_grid(
                x_values=np.asarray(test_scores, dtype=float),
                y_values=np.asarray(hamd_true, dtype=float),
                groups=session_labels,
                output_path=session_grid_path,
                title="EPI: Predicted continuous score (0-1) vs True HAM-D per session",
                x_label="Predicted continuous score (0-1)",
            )
            if n_session_panels > 0:
                print(f"Saved EPI session cloud grid ({n_session_panels} panels): {session_grid_path}")

            trajectory_grid_path = os.path.join(PERFORMANCE_DIR, "epi_subject_trajectories_grid.png")
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



