# backend/scripts/train_model.py

import os
import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset, DatasetDict

import torch
print("CUDA available:", torch.cuda.is_available(), " | Device count:", torch.cuda.device_count())


# ─── 1) Load & Sample CSVs ────────────────────────────────────────────────
train_df = pd.read_csv("data/raw/train.csv")
test_df  = pd.read_csv("data/raw/test.csv")

# sample down for local dev
MAX_TRAIN = 100_000
MAX_TEST  =  20_000
train_df = train_df.sample(n=min(len(train_df), MAX_TRAIN), random_state=42)
test_df  = test_df.sample(n=min(len(test_df),  MAX_TEST), random_state=42)
print(f"Using {len(train_df)} training samples and {len(test_df)} testing samples")

# convert to HuggingFace Datasets
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
    "test":  Dataset.from_pandas(test_df.reset_index(drop=True)),
})
del train_df, test_df
gc.collect()

# ─── 2) Tokenization ────────────────────────────────────────────────────
MODEL_NAME = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

dataset = dataset.map(
    tokenize_fn,
    batched=True,
    batch_size=500,
    remove_columns=["text"],
)
dataset.set_format("torch")

# ─── 3) Model Initialization ─────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)

# ─── 4) Training Configuration ──────────────────────────────────────────
# Calculate how many steps per epoch (for save/eval)
steps_per_epoch = max(1, len(dataset["train"]) // 16)

# training_args = TrainingArguments(
#     output_dir="models/sentiment",
#     num_train_epochs=3,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=32,
#     logging_steps=500,
#     save_steps=steps_per_epoch,   # checkpoint at end of each epoch
#     eval_steps=steps_per_epoch,   # eval at end of each epoch
#     save_total_limit=2,           # keep only last 2 checkpoints
#     # load_best_model_at_end removed to avoid strategy mismatch
# )

training_args = TrainingArguments(
    output_dir="models/sentiment",
    num_train_epochs=3,
    per_device_train_batch_size=8,    # lower this if you hit OOM
    per_device_eval_batch_size=16,
    logging_steps=500,
    save_steps=steps_per_epoch,
    eval_steps=steps_per_epoch,
    save_total_limit=2,

    # NEW GPU-ACCELERATION FLAGS:
    fp16=True,                        # use mixed precision
    gradient_accumulation_steps=2,    # simulate batch size of 16
    dataloader_num_workers=4,         # parallelize data loading
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)

# ─── 5) MLflow Tracking & Training ──────────────────────────────────────
mlflow.set_experiment("sentiment-reco")
with mlflow.start_run():
    mlflow.log_param("model_name", MODEL_NAME)
    mlflow.log_param("train_samples", len(dataset["train"]))
    mlflow.log_param("test_samples",  len(dataset["test"]))
    mlflow.log_param("epochs",       training_args.num_train_epochs)
    mlflow.log_param("batch_size",   training_args.per_device_train_batch_size)

    # 5a) Train + checkpoint
    trainer.train()

    # 5b) Final evaluation on test set
    metrics = trainer.evaluate()
    mlflow.log_metrics(metrics)

    # 5c) Save final model & tokenizer
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    mlflow.log_artifacts(training_args.output_dir, artifact_path="sentiment_model")

print("✅ Training complete. Model saved to models/sentiment/")
