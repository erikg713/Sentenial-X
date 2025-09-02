"""
model_utils.py
---------------
Production-ready utilities for managing ML models in Sentenial-X.
Supports PyTorch & ONNX models, checkpointing, preprocessing, metrics, and model export.
"""

import os
import logging
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import numpy as np
import onnx

# Logger setup
logger = logging.getLogger("sentenialx.ml.model_utils")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ----------------------------
# PyTorch Model Utilities
# ----------------------------
def save_model(model: nn.Module, path: str):
    """Save PyTorch model state_dict."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"Saved PyTorch model at {path}")


def load_model(model: nn.Module, path: str, device: str = "cpu") -> nn.Module:
    """Load PyTorch model state_dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    logger.info(f"Loaded PyTorch model from {path} to {device}")
    return model


def export_to_onnx(model: nn.Module, input_sample: torch.Tensor, onnx_path: str, opset_version: int = 14):
    """Export PyTorch model to ONNX format."""
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    torch.onnx.export(
        model,
        input_sample,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    # Validate ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    logger.info(f"Exported model to ONNX: {onnx_path}")


# ----------------------------
# Data Preprocessing Utilities
# ----------------------------
def to_tensor(ndarray: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Convert numpy array to torch tensor."""
    return torch.tensor(ndarray, dtype=torch.float32, device=device)


def batchify(data: np.ndarray, batch_size: int):
    """Yield batches from numpy array."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


# ----------------------------
# Metrics Utilities
# ----------------------------
def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy."""
    correct = (preds.argmax(dim=1) == labels).sum().item()
    total = labels.size(0)
    return correct / total


def mse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute mean squared error."""
    return nn.functional.mse_loss(preds, targets).item()


# ----------------------------
# Misc Utilities
# ----------------------------
def ensure_dir(path: str):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)
    logger.debug(f"Ensured directory exists: {path}")


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Dummy PyTorch model
    class DummyModel(nn.Module):
        def __init__(self, input_dim=10, output_dim=2):
            super().__init__()
            self.fc = nn.Linear(input_dim, output_dim)

        def forward(self, x):
            return self.fc(x)

    model = DummyModel()
    sample_input = torch.randn(1, 10)

    save_path = "checkpoints/dummy_model.pt"
    onnx_path = "checkpoints/dummy_model.onnx"

    save_model(model, save_path)
    load_model(model, save_path)
    export_to_onnx(model, sample_input, onnx_path)

    # Dummy metrics
    labels = torch.tensor([1])
    preds = torch.tensor([[0.1, 0.9]])
    print("Accuracy:", accuracy(preds, labels))
    print("MSE:", mse(preds, labels.float()))
