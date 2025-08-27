# ===== File: core/detection/__init__.py =====
"""Detection subsystem for Sentenial‑X.

Handles signature-based, rule-based, and ML-based detection in a unified pipeline.
"""
from .base import DetectionEvent, Detector, DetectionVerdict, Severity
from .pipeline import DetectionPipeline
from .signatures import SignatureEngine
from .rules import RuleEngine
from .ml import MLClassifier
from .config import DetectionConfig

__all__ = [
    "DetectionEvent",
    "Detector",
    "DetectionVerdict",
    "Severity",
    "DetectionPipeline",
    "SignatureEngine",
    "RuleEngine",
    "MLClassifier",
    "DetectionConfig",
]
