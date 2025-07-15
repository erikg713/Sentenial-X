import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

class Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, batch_size=32, lr=1e-3, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.criterion = CrossEntropyLoss()

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(inputs), labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate(self):
        if not self.val_loader:
            return None
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total if total > 0 else 0
        return total_loss / len(self.val_loader), accuracy

    def fit(self, epochs=10):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_result = self.validate()
            if val_result:
                val_loss, val_acc = val_result
                print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            else:
                print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}")
