# libs/ml/pytorch/trainer.py

import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from tqdm import tqdm
from typing import List, Optional, Tuple
from .model import TelemetryModel
from transformers import AutoTokenizer


# ---------------------------
# Dataset wrapper for supervised or contrastive training
# ---------------------------
class TelemetryTrainerDataset(Dataset):
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None, tokenizer_name: str = "bert-base-uncased", max_len: int = 128, contrastive: bool = False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.contrastive = contrastive

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in encoded.items()}

        if self.contrastive:
            # Simulated contrastive view (can replace with real augmentations)
            view2 = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
            item2 = {k: v.squeeze(0) for k, v in view2.items()}
            return item, item2

        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------
# Contrastive loss (SimCLR style)
# ---------------------------
def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim_matrix = torch.matmul(z, z.T) / temperature
    labels = torch.arange(batch_size).to(z.device)
    labels = torch.cat([labels, labels], dim=0)

    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    sim_matrix = sim_matrix.masked_fill(mask, -9e15)

    return nn.CrossEntropyLoss()(sim_matrix, labels)


# ---------------------------
# Trainer class
# ---------------------------
class TrainerModule:
    def __init__(self, model: TelemetryModel, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def train_supervised(
        self,
        texts: List[str],
        labels: List[int],
        batch_size: int = 16,
        epochs: int = 3,
        lr: float = 5e-5
    ):
        dataset = TelemetryTrainerDataset(texts, labels=labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels_tensor = batch["labels"].to(self.device)
                outputs = self.model(input_ids, attention_mask, labels=labels_tensor)
                loss = outputs["loss"]
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1} Loss: {epoch_loss/len(dataloader):.4f}")

    def train_contrastive(
        self,
        texts: List[str],
        batch_size: int = 16,
        epochs: int = 5,
        lr: float = 3e-4,
        temperature: float = 0.5
    ):
        dataset = TelemetryTrainerDataset(texts, contrastive=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for view1, view2 in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                input_ids1 = view1["input_ids"].to(self.device)
                attention_mask1 = view1["attention_mask"].to(self.device)
                input_ids2 = view2["input_ids"].to(self.device)
                attention_mask2 = view2["attention_mask"].to(self.device)

                z1 = self.model(input_ids1, attention_mask1)
                z2 = self.model(input_ids2, attention_mask2)
                loss = nt_xent_loss(z1, z2, temperature)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1} Loss: {epoch_loss/len(dataloader):.4f}")


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    from .model import TelemetryModel

    sample_texts = [
        "Agent executed task A",
        "Telemetry anomaly detected",
        "Normal telemetry received"
    ]
    sample_labels = [0, 1, 0]

    model = TelemetryModel(embedding_dim=64, num_classes=2)
    trainer = TrainerModule(model)

    # Supervised training
    trainer.train_supervised(sample_texts, sample_labels, epochs=1)

    # Contrastive training
    trainer.train_contrastive(sample_texts, epochs=1)
