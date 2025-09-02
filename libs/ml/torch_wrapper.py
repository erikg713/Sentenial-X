"""
TorchWrapper
-------------
A production-ready PyTorch wrapper for Sentenial-X ML models.
Provides training, inference, device management, and logging support.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Callable, Any
from datetime import datetime
import logging

# Initialize logger
logger = logging.getLogger("sentenialx.ml.torch")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class TorchWrapper:
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        loss_fn: Optional[Callable] = None,
        optimizer_fn: Optional[Callable] = None,
        lr: float = 1e-3,
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.optimizer = optimizer_fn(self.model.parameters(), lr=lr) if optimizer_fn else optim.Adam(self.model.parameters(), lr=lr)
        logger.info(f"TorchWrapper initialized on device: {self.device}")

    def train(
        self,
        dataloader: DataLoader,
        epochs: int = 10,
        val_dataloader: Optional[DataLoader] = None,
        checkpoint_path: Optional[str] = None,
    ):
        logger.info(f"Starting training for {epochs} epochs")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

            if val_dataloader:
                val_loss, val_acc = self.evaluate(val_dataloader)
                logger.info(f"Validation - Loss: {val_loss:.6f}, Accuracy: {val_acc:.2f}%")

            if checkpoint_path:
                torch.save(self.model.state_dict(), f"{checkpoint_path}_epoch{epoch+1}.pt")
                logger.info(f"Checkpoint saved: {checkpoint_path}_epoch{epoch+1}.pt")

    def evaluate(self, dataloader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def predict(self, inputs: torch.Tensor) -> Any:
        self.model.eval()
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs

    def load_checkpoint(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Checkpoint loaded: {path}")

    def save_checkpoint(self, path: str):
        torch.save(self.model.state_dict(), path)
        logger.info(f"Checkpoint saved: {path}")


# Example Dataset for testing
class DummyDataset(Dataset):
    def __init__(self, size=1000, input_dim=10, num_classes=2):
        self.X = torch.randn(size, input_dim)
        self.y = torch.randint(0, num_classes, (size,))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Example usage
if __name__ == "__main__":
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    wrapper = TorchWrapper(model)
    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    wrapper.train(dataloader, epochs=2)
    sample = torch.randn(5, 10)
    preds = wrapper.predict(sample)
    print(preds)
