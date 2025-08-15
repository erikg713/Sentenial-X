# models/encoder/traffic_encoder.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from typing import List, Optional
from transformers import AutoTokenizer, AutoModel

class TrafficDataset(Dataset):
    """
    Wraps HTTP/network sequences for transformer encoder.
    """
    def __init__(self, sequences: List[str], tokenizer: AutoTokenizer, max_length: int = 128):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = self.tokenizer(sequences, truncation=True, padding=True, max_length=max_length)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}

class TrafficEncoder:
    """
    Encodes HTTP/network traffic into fixed-size embeddings using transformer model.
    """
    def __init__(self, model_name: str = "bert-base-uncased", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def encode(self, sequences: List[str], batch_size: int = 16):
        dataset = TrafficDataset(sequences, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        embeddings = []
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # Take [CLS] token embedding
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings.append(cls_embeddings.cpu())
        return torch.cat(embeddings, dim=0)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"[TrafficEncoder] Model saved at {path}")

    def load(self, path: str):
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(self.device)
        print(f"[TrafficEncoder] Model loaded from {path}")
