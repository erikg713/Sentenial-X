# sentenial-x/ai_core/__init__.py
"""
Sentenial-X AI Core
------------------
Central hub for AI-driven functionality:
- Threat classification (ML/LLM)
- Jailbreak detection (NLP)
- Log and telemetry encoding (embeddings)
- Countermeasure prediction
- Integration with EndpointAgent and RetaliationBot
"""

from .orchestrator import AICoreOrchestrator
from .threat_classifier import ThreatClassifier
from .encoder import ThreatTextEncoder
from .jailbreak_detector import JailbreakDetector
from .countermeasure_predictor import CountermeasurePredictor
