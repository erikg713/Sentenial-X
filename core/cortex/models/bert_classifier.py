#!/usr/bin/env python3
"""
Sentenial-X :: BERT Threat Classifier
=====================================

Purpose:
    Fine-tune or infer with BERT for semantic threat classification
    (e.g., threat intents, zero-day patterns, anomaly labeling).

Design:
    - HuggingFace Transformers integration
    - PyTorch backend
    - Tokenization & batching utilities
    - Inference & probability outputs
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from typing import List, Optional, Dict


# ------------------------------------------------------------
# Dataset Wrapper
# ------------------------------------------------------------
class ThreatDataset(Dataset):
    """
    Torch Dataset wrapper for text -> label data
    """

    def __init__(self, texts: List[str], labels: Optional[List[int]] = None, tokenizer_name: str = "bert-base-uncased", max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ------------------------------------------------------------
# BERT Classifier
# ------------------------------------------------------------
class BertClassifier:
    """
    Thin wrapper for BERT-based classification
    """

    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 2, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model.to(self.device)
        self.model.eval()  # default inference mode

        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def predict(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, float]]:
        dataset = ThreatDataset(texts, tokenizer_name=None)
        loader = DataLoader(dataset, batch_size=batch_size)

        results = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)

                for prob in probs:
                    results.append({str(i): float(prob[i]) for i in range(prob.size(0))})
        return results

    def fine_tune(self, train_texts: List[str], train_labels: List[int], epochs: int = 3, lr: float = 5e-5, batch_size: int = 16):
        """
        Optional: Fine-tune the model on custom dataset
        """
        dataset = ThreatDataset(train_texts, train_labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")

        self.model.eval()

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str):
        self.model = BertForSequenceClassification.from_pretrained(path)
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model.to(self.device)
        self.model.eval()


# ------------------------------------------------------------
# CLI interface (optional)
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="BERT Threat Classifier CLI")
    parser.add_argument("--texts", required=True, help="JSON array of texts to classify")
    parser.add_argument("--model_path", default=None, help="Path to pre-trained model")
    args = parser.parse_args()

    texts = json.loads(args.texts)
    classifier = BertClassifier()
    if args.model_path:
        classifier.load(args.model_path)

    predictions = classifier.predict(texts)
    print(json.dumps(predictions, indent=4))
