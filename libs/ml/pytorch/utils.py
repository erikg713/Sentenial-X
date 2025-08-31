# libs/ml/pytorch/utils.py

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import List, Optional
import os


# ---------------------------
# Device utility
# ---------------------------
def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Returns the available device ('cuda' if available and preferred, else 'cpu').
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------
# Metrics utilities
# ---------------------------
def compute_classification_metrics(preds: List[int], labels: List[int]) -> dict:
    """
    Compute common classification metrics: accuracy, precision, recall, F1-score
    """
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="binary"),
        "recall": recall_score(labels, preds, average="binary"),
        "f1": f1_score(labels, preds, average="binary")
    }
    return metrics


def evaluate_model(model, dataloader, device: Optional[torch.device] = None):
    """
    Evaluate a supervised model on a dataloader
    """
    device = device or get_device()
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask, labels=labels)
            preds = torch.argmax(outputs["logits"], dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return compute_classification_metrics(all_preds, all_labels)


# ---------------------------
# Batch embedding utility
# ---------------------------
def batch_encode_texts(model, texts: List[str], tokenizer, batch_size: int = 32, device: Optional[torch.device] = None, max_len: int = 128):
    """
    Encode texts into embeddings in batches
    """
    device = device or get_device()
    model.to(device)
    model.eval()
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            emb = model(input_ids, attention_mask)
            embeddings.append(emb.cpu())

    return torch.vstack(embeddings).numpy()


# ---------------------------
# File utilities
# ---------------------------
def ensure_dir(path: str):
    """
    Ensure that the directory exists
    """
    os.makedirs(path, exist_ok=True)


# ---------------------------
# Simple logging
# ---------------------------
def log(message: str):
    """
    Print a standardized log message
    """
    print(f"[ML-UTILS] {message}")


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    log("Testing utils module...")

    # Device test
    device = get_device()
    log(f"Using device: {device}")

    # Metrics test
    preds = [0, 1, 0, 1]
    labels = [0, 1, 1, 1]
    metrics = compute_classification_metrics(preds, labels)
    log(f"Metrics: {metrics}")
