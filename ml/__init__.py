# ml/__init__.py

"""
Sentenial-X ML package

Provides:
- ML pipelines for telemetry/log analysis
- BERT-based classifiers
- LoRA fine-tuning support
- Supervised and contrastive datasets
- Utilities for training, evaluation, and embeddings
"""

from .ml_pipeline import SentenialMLPipeline
from .train_bert_intent_classifier import TelemetryDataset, TelemetryContrastiveDataset

__all__ = [
    "SentenialMLPipeline",
    "TelemetryDataset",
    "TelemetryContrastiveDataset",
]
