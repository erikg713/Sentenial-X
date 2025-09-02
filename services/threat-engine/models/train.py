import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from pytorch_classifiers import SimpleThreatClassifier

# Dummy training setup (replace with real features + labels later)
def get_dummy_data(n_samples=1000, input_dim=20, num_classes=2):
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, num_classes, (n_samples,))
    dataset = TensorDataset(X, y)
    return dataset

def train_model():
    input_dim = 20
    num_classes = 2
    model = SimpleThreatClassifier(input_dim=input_dim, num_classes=num_classes)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dataset = get_dummy_data(input_dim=input_dim, num_classes=num_classes)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(5):  # small loop for example
        total_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Save trained model
    model_path = Path(__file__).parent / "threat_model.pkl"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
