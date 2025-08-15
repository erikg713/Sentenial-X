# sentenialx/ml/train_bert_intent_classifier.py
import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import random

# ---------------- Hyperparameters ----------------
MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "./output/bert_intent_classifier"
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5
MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- Sample Dataset ----------------
# Replace with your real CVE logs, payloads, HTTP requests, etc.
samples = [
    ("User login failed from IP 192.168.1.10", 0),       # normal
    ("DROP TABLE users; -- SQL Injection", 1),           # sql_injection
    ("<script>alert('XSS')</script>", 2),               # xss
    ("File download detected: /tmp/malware.exe", 3),    # malware
    ("POST /api/upload HTTP/1.1 Content-Length: 1024", 0),
] * 200  # replicate for demonstration

texts, labels = zip(*samples)
labels = list(labels)

# ---------------- Train/Test Split ----------------
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.15, random_state=42
)

# ---------------- Dataset ----------------
class ThreatDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# ---------------- Tokenizer & Model ----------------
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
train_dataset = ThreatDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = ThreatDataset(val_texts, val_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)
model.to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)

# ---------------- Training Loop ----------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_loss:.4f}")

    # ---------------- Validation ----------------
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            preds.extend(predictions.cpu().numpy())
            true.extend(labels.cpu().numpy())
    print(classification_report(true, preds, digits=4))

# ---------------- Save Model ----------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"BERT intent classifier saved to {OUTPUT_DIR}")    logging_dir="./logs",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
)

trainer.train()
model.save_pretrained("bert-threat-intent")
tokenizer.save_pretrained("bert-threat-intent")
