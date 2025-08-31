# src/encoder_training.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional


class TelemetryDataset(Dataset):
    """
    Custom dataset for telemetry/log text data and optional labels.
    """

    def __init__(self, texts: List[str], labels: Optional[List[int]] = None, tokenizer_name: str = "distilbert-base-uncased", max_len: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoded.items()}
        if self.labels:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class TransformerEncoder(nn.Module):
    """
    Transformer-based encoder for generating embeddings.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", embedding_dim: int = 128):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.embedding_dim = embedding_dim
        self.projection = nn.Linear(self.transformer.config.hidden_size, embedding_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # CLS token
        projected = self.projection(cls_emb)
        return projected


def train_encoder(
    texts: List[str],
    labels: Optional[List[int]] = None,
    model_name: str = "distilbert-base-uncased",
    embedding_dim: int = 128,
    batch_size: int = 16,
    epochs: int = 3,
    lr: float = 1e-4,
    device: Optional[str] = None,
    save_path: Optional[str] = None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TelemetryDataset(texts, labels, tokenizer_name=model_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TransformerEncoder(model_name, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss() if labels else nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            optimizer.zero_grad()

            embeddings = model(input_ids, attention_mask)

            if labels:
                labels_tensor = batch["label"].to(device)
                loss = criterion(embeddings, labels_tensor)
            else:
                # Self-supervised: try to reconstruct or use contrastive loss
                loss = embeddings.norm(p=2, dim=1).mean()  # dummy L2 loss

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(dataloader):.4f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    return model


def evaluate_encoder(model: TransformerEncoder, texts: List[str], device: Optional[str] = None) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model.transformer.name_or_path)
    model.to(device)
    model.eval()
    embeddings = []

    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            emb = model(input_ids, attention_mask)
            embeddings.append(emb.cpu().numpy())

    return np.vstack(embeddings)


# Example usage
if __name__ == "__main__":
    sample_texts = ["Agent executed task A", "Telemetry event received", "Anomaly detected"]
    model = train_encoder(sample_texts, embedding_dim=64, epochs=1, save_path="models/telemetry_encoder.pt")
    emb_vectors = evaluate_encoder(model, ["New telemetry log"])
    print("Embedding shape:", emb_vectors.shape)
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from torch import nn
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

class HTTPEncoder(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.proj = nn.Linear(base.config.hidden_size, 256)
    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids, attention_mask=attention_mask).last_hidden_state
        cls = outputs[:,0]
        return self.proj(cls)

# Dataset loading & mapping to input_ids/labels
# Train to classify malicious vs. benign sessions

# ... Trainer boilerplate ...
