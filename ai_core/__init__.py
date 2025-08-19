# sentenial-x/ai_core/__init__.py

"""
Sentenial-X AI Core
-------------------
Centralized interface for AI-driven security intelligence:

- Threat classification (ML/LLM)
- Jailbreak detection (NLP)
- Log and telemetry encoding (embeddings)
- Countermeasure prediction
- Integration points for EndpointAgent and RetaliationBot
"""

from .orchestrator import AICoreOrchestrator
from .threat_classifier import ThreatClassifier
from .encoder import ThreatTextEncoder
from .jailbreak_detector import JailbreakDetector
from .countermeasure_predictor import CountermeasurePredictor

__all__ = [
    "AICoreOrchestrator",
    "ThreatClassifier",
    "ThreatTextEncoder",
    "JailbreakDetector",
    "CountermeasurePredictor",
]
