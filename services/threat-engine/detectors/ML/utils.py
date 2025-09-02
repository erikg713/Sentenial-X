import torch
import os
import numpy as np
from torch.utils.data import TensorDataset

def save_model(model: torch.nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model: torch.nn.Module, path: str) -> torch.nn.Module:
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
    return model

def load_dataset(path: str):
    """
    Load dataset for training. Expects npz file with 'features' and 'labels'.
    """
    import numpy as np
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    data = np.load(path)
    features = torch.tensor(data['features'], dtype=torch.float32)
    labels = torch.tensor(data['labels'], dtype=torch.long)
    return TensorDataset(features, labels)

def evaluate_model(model: torch.nn.Module, dataloader, device: str):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            predicted = torch.argmax(outputs, dim=1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    accuracy = correct / total if total > 0 else 0.0
    print(f"Evaluation Accuracy: {accuracy*100:.2f}%")
