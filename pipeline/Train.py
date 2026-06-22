import os
import gc
import csv
import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, DataCollatorForLanguageModeling, AutoModelForMaskedLM
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
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
PLOT_DIR = "/zi/home/luke.bregulla/Desktop/DSS/results/Validation/performance_plots"
PER_SUBJECT_PREDICTIONS = os.path.join(OUTPUT_DIR, "validation_subject_predictions_V12.csv")
BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model")
DISTRIBUTION_PLOT = os.path.join(PLOT_DIR, "training_class_distribution_train.png")


os.environ["TENSORBOARD_LOGGING_DIR"] = LOGGING_DIR

# model specifics
BASE_MODEL = "FacebookAI/xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
DOMAIN_ADAPTED_MODEL_DIR = os.path.join(OUTPUT_DIR, "domain_adapted_bert")
SAVE_PLOTS = True


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Config
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
EFN_WEIGHT = 1.0


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Functions and Helpers
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Class-balanced weights using effective number weightin
def build_class_weights(
    target_classes,
    num_classes=NUM_CLASSES,
    beta=CLASS_BALANCE_BETA,
    power=EFN_WEIGHT,
):
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



#Balanced accuracy 3-class label space
def balanced_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).astype(float)
    recalls = []
    for i in range(3):
        denom = cm[i, :].sum()
        recalls.append((cm[i, i] / denom) if denom > 0 else 0.0)
    return float(np.mean(recalls))


#overall metrics csv
def overall_metrics(output_path, metrics_row):
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


#per subject metrics
def subject_predictions(output_path, row_ids, true_classes, pred_classes, pred_probs):
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


#plot class distribution
def distribution_plot(counts, output_path):
    labels = ["none", "moderate", "severe"]
    counts = np.asarray(counts, dtype=int)
    
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x = np.arange(len(labels), dtype=float)
    bars = ax.bar(x, counts, width=0.62, color="#4C78A8", alpha=0.9)
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.5, f"{int(h)}", 
                ha="center", va="bottom", fontsize=9)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of cases")
    ax.set_title("Training Dataset Class Distribution")
    ax.grid(axis="y", alpha=0.25)
    
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)






#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Dataset loading and splitting
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

train_dataset = load_dataset("csv", data_files={"train": TRAIN_FILE})["train"]
train_labels = np.asarray(train_dataset["labels"], dtype=np.int64)
train_class_counts = np.bincount(train_labels, minlength=3).astype(int)

distribution_plot(train_class_counts, DISTRIBUTION_PLOT)
print(f"Saved training dataset class distribution plot: {DISTRIBUTION_PLOT}")
print(
    "Training class distribution (none / moderate / severe): "
    f"{train_class_counts.tolist()}"
)


# Split into train/validate 80/20
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

for split_name in ["train", "validation"]:
    dataset[split_name] = dataset[split_name].map(lambda x: {"target_class": int(x["labels"])})

train_classes = np.asarray(dataset["train"]["target_class"], dtype=np.int64)
class_weights_np, class_counts = build_class_weights(train_classes)
print(f"  Class counts [0,1,2]: {class_counts.tolist()}")
print(f"  Class weights [0,1,2]: {[round(float(w), 4) for w in class_weights_np.tolist()]}")


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Custom Models
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# chunk level encoding
# mean pooling
# 3-class head
class CustomBERTModel(nn.Module):
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


        if use_gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

    #encode sequentially to reduce memory use
    def _encode_chunks_sequential(self, input_ids, attention_mask):

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


# Custom collator to handle chunking and subject-level aggregation
class SubjectChunkingCollator:
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
# Metrics
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def extract_pred_classes(predictions):
    logits = np.asarray(predictions[0] if isinstance(predictions, (tuple, list)) else predictions, dtype=float)
    if logits.ndim == 1:
        logits = logits.reshape(-1, NUM_CLASSES)
    return np.argmax(logits, axis=1).astype(np.int64)


def extract_pred_probabilities(predictions):
    logits = np.asarray(predictions[0] if isinstance(predictions, (tuple, list)) else predictions, dtype=float)
    if logits.ndim == 1:
        logits = logits.reshape(-1, NUM_CLASSES)
    logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
    return np.exp(logits_shifted) / np.sum(np.exp(logits_shifted), axis=1, keepdims=True)


def compute_metrics(eval_pred):
    predictions, true_classes = eval_pred
    pred_classes = extract_pred_classes(predictions)
    true_classes = np.asarray(true_classes, dtype=np.int64).reshape(-1)
    return {
        "class_accuracy": np.mean(pred_classes == true_classes),
        "balanced_accuracy": balanced_accuracy(true_classes, pred_classes),
        "macro_f1": f1_score(true_classes, pred_classes, average="macro", zero_division=0),
    }

#check if model is already saved
def model_weights(model_dir):
    """Check if model checkpoint exists with both config and weights."""
    return (os.path.isfile(os.path.join(model_dir, "config.json")) and
            (os.path.isfile(os.path.join(model_dir, "model.safetensors")) or
             os.path.isfile(os.path.join(model_dir, "pytorch_model.bin"))))

#load checkpoint weights into model 
def load_checkpoint(model, checkpoint_dir):
    safe_path = os.path.join(checkpoint_dir, "model.safetensors")
    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    
    if os.path.isfile(safe_path):
        state_dict = load_safetensors(safe_path)
    elif os.path.isfile(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        return False
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    for key_type, keys in [("missing", missing), ("unexpected", unexpected)]:
        if keys:
            print(f"Checkpoint load warning: {key_type} keys={len(keys)}")
    return True



#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Pre-Training Domain Adaptation (MLM)
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

# if no model detected, run MLM domain adaptation on training data
if not model_weights(DOMAIN_ADAPTED_MODEL_DIR):
    daic_text = dataset["train"].remove_columns([c for c in dataset["train"].column_names if c != "text"])
    print(f"samples (DAIC: {len(daic_text)})")

    mlm_tokenized_dataset = daic_text.map(
        tokenize_mlm,
        batched=True,
        remove_columns=["text"],
    )

    mlm_language_model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)
    mlm_language_model.config.tie_word_embeddings = False
    mlm_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    mlm_model_trainer = Trainer(
        model=mlm_language_model,
        args=mlm_args,
        train_dataset=mlm_tokenized_dataset,
        data_collator=mlm_data_collator,
    )

    mlm_model_trainer.train()
    mlm_model_trainer.save_model(DOMAIN_ADAPTED_MODEL_DIR)
    print(f"Saved domain-adapted checkpoint to: {DOMAIN_ADAPTED_MODEL_DIR}")

    # Free MLM stage GPU memory before classifier fine-tuning.
    del mlm_model_trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    del mlm_language_model
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

# Check for existing model before training
if os.path.isdir(BEST_MODEL_DIR) and load_checkpoint(trainer.model, BEST_MODEL_DIR):
    print(f"Skipping classifier training: loaded best_model from {BEST_MODEL_DIR}")
else:
    if os.path.isdir(BEST_MODEL_DIR):
        print("Best model found but load failed; running classifier training.")
    else:
        print("No best_model found; training classifier.")
    trainer.train(resume_from_checkpoint=False)


# Validation
validation_predictions = trainer.predict(dataset["validation"])
validation_predicted_classes = extract_pred_classes(validation_predictions.predictions)
validation_predicted_probabilities = extract_pred_probabilities(validation_predictions.predictions)
validation_true_classes = np.asarray(dataset["validation"]["target_class"], dtype=np.int64)
validation_row_ids = [str(i) for i in range(len(dataset["validation"]))]

validation_accuracy = np.mean(validation_predicted_classes == validation_true_classes)
validation_balanced_accuracy = balanced_accuracy(validation_true_classes, validation_predicted_classes)
validation_macro_f1 = f1_score(validation_true_classes, validation_predicted_classes, average="macro", zero_division=0)

# Report results
best_checkpoint = trainer.state.best_model_checkpoint

print("\n" + "="*60)
print("VALIDATION RESULTS")
print("="*60)
print(f"Accuracy:          {validation_accuracy:.4f}")
print(f"Balanced Accuracy: {validation_balanced_accuracy:.4f}")
print(f"Macro F1:          {validation_macro_f1:.4f}")

validation_confusion_matrix = confusion_matrix(validation_true_classes, validation_predicted_classes, labels=[0, 1, 2])
print("Confusion Matrix (rows=true, cols=pred):")
print(validation_confusion_matrix)

subject_predictions(
    PER_SUBJECT_PREDICTIONS,
    row_ids=validation_row_ids,
    true_classes=validation_true_classes,
    pred_classes=validation_predicted_classes,
    pred_probs=validation_predicted_probabilities,
)
print(f"Saved validation per-subject prediction CSV: {PER_SUBJECT_PREDICTIONS}")

# Save overall validation metrics
validation_metrics_path = os.path.join(OUTPUT_DIR, "validation_overall_metrics.csv")
overall_metrics(
    validation_metrics_path,
    {
        "n": len(validation_true_classes),
        "accuracy": validation_accuracy,
        "balanced_accuracy": validation_balanced_accuracy,
        "macro_f1": validation_macro_f1,
    },
)
print(f"Saved validation overall metrics CSV: {validation_metrics_path}")

if SAVE_PLOTS:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(validation_confusion_matrix, cmap="Blues")
    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["none", "moderate", "severe"])
    ax.set_yticklabels(["none", "moderate", "severe"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Validation")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(validation_confusion_matrix[i, j]), ha="center", va="center")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    validation_cm_path = os.path.join(PLOT_DIR, "validation_confusion_matrix.png")
    plt.savefig(validation_cm_path, dpi=180)
    plt.close()
    print(f"Saved validation confusion matrix: {validation_cm_path}")

