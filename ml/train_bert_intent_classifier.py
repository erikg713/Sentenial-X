# ml/train_bert_intent_classifier.py

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from typing import List

from libs.ml.pytorch.dataset import TelemetryDataset
from libs.ml.pytorch.model import TelemetryModel
from libs.ml.pytorch.trainer import TrainerModule
from libs.ml.pytorch.utils import get_device, log, compute_classification_metrics
from libs.ml.pytorch.lora_tuner import LoRATuner


# ---------------------------
# Sample telemetry/log dataset
# ---------------------------
texts = [
    "Agent executed task A",
    "Telemetry anomaly detected",
    "Normal telemetry received",
    "VSSAdmin delete shadows detected",
    "Process injected with AMSI bypass"
]

# Labels: 0 = benign, 1 = malicious
labels = [0, 1, 0, 1, 1]


# ---------------------------
# Hyperparameters
# ---------------------------
BASE_MODEL = "bert-base-uncased"
EMBEDDING_DIM = 128
NUM_CLASSES = 2
BATCH_SIZE = 2
EPOCHS = 3
LR = 5e-5
MAX_LEN = 128
USE_LORA = True  # <-- enable LoRA fine-tuning
LORA_R = 4
LORA_ALPHA = 8
LORA_DROPOUT = 0.1
LORA_SAVE_PATH = "models/lora_bert_intent.pt"


# ---------------------------
# Device
# ---------------------------
device = get_device()
log(f"Using device: {device}")


# ---------------------------
# Dataset & DataLoader
# ---------------------------
dataset = TelemetryDataset(texts, labels, tokenizer_name=BASE_MODEL, max_len=MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# ---------------------------
# Model initialization
# ---------------------------
if USE_LORA:
    # Initialize LoRA tuner
    lora_tuner = LoRATuner(base_model_name=BASE_MODEL, num_labels=NUM_CLASSES, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT, device=device)
    model = lora_tuner.model
    trainer = None  # Training is handled via LoRA tuner
    log("Initialized LoRA tuner for BERT classifier.")
else:
    # Standard full model training
    model = TelemetryModel(base_model_name=BASE_MODEL, embedding_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES)
    trainer = TrainerModule(model, device=device)
    log("Initialized standard BERT classifier.")


# ---------------------------
# Training loop
# ---------------------------
if USE_LORA:
    log("Starting LoRA fine-tuning...")
    lora_tuner.train(texts, labels, batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR, save_path=LORA_SAVE_PATH)
    log(f"LoRA fine-tuning completed. Model saved to {LORA_SAVE_PATH}")
else:
    log("Starting standard supervised training...")
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        model.train()
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_tensor = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask, labels=labels_tensor)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        log(f"Epoch {epoch + 1} average loss: {epoch_loss / len(dataloader):.4f}")

    # Save standard model
    torch.save(model.state_dict(), "models/bert_intent_classifier.pt")
    log("Standard BERT classifier saved to models/bert_intent_classifier.pt")


# ---------------------------
# Evaluation
# ---------------------------
log("Evaluating model...")
all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_tensor = batch["labels"].to(device)

        if USE_LORA:
            logits = lora_tuner.model(input_ids, attention_mask, labels=None)
        else:
            logits = model(input_ids, attention_mask, labels=None)
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_tensor.cpu().numpy())

metrics = compute_classification_metrics(all_preds, all_labels)
log(f"Evaluation metrics: {metrics}")


# ---------------------------
# Inference example
# ---------------------------
new_texts = ["Suspicious AMSI bypass detected", "Routine telemetry event"]
model.eval()
predictions = []

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
with torch.no_grad():
    for text in new_texts:
        encoded = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        if USE_LORA:
            logits = lora_tuner.model(input_ids, attention_mask, labels=None)
        else:
            logits = model(input_ids, attention_mask, labels=None)
        preds = torch.argmax(logits, dim=-1)
        predictions.append(preds.item())

log(f"Inference predictions: {predictions}")
