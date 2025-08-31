# libs/ml/pytorch/__init__.py

"""
PyTorch ML module for Sentenial-X telemetry/log processing.

Provides:
- TelemetryModel: Transformer-based embedding and classification model
- TrainerModule: Supervised and contrastive training
- Datasets: Supervised and contrastive PyTorch datasets
- Utils: Device, metrics, logging, batch encoding helpers
"""

from .model import TelemetryModel, encode_texts
from .trainer import TrainerModule
from .dataset import TelemetryDataset, TelemetryContrastiveDataset, get_supervised_dataset, get_contrastive_dataset
from .utils import get_device, compute_classification_metrics, evaluate_model, batch_encode_texts, ensure_dir, log

__all__ = [
    "TelemetryModel",
    "encode_texts",
    "TrainerModule",
    "TelemetryDataset",
    "TelemetryContrastiveDataset",
    "get_supervised_dataset",
    "get_contrastive_dataset",
    "get_device",
    "compute_classification_metrics",
    "evaluate_model",
    "batch_encode_texts",
    "ensure_dir",
    "log"
]
