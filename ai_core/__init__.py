# -*- coding: utf-8 -*-
"""
AI Core Package Initializer for Sentenial-X
-------------------------------------------

Provides access to core AI and ML functionality, including:
- NLP-based threat analysis
- Adversarial input detection
- Predictive threat modeling
"""

from __future__ import annotations

import logging

from .nlp_analyzer import NLPAnalyzer
from .adversarial_detector import AdversarialDetector
from .predictive_model import PredictiveThreatModel

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = [
    "NLPAnalyzer",
    "AdversarialDetector",
    "PredictiveThreatModel",
    "logger",
]

# ---------------------------------------------------------------------------
# Initialize core AI components (lazy load to save resources)
# ---------------------------------------------------------------------------

_ai_instances = {}

def get_nlp_analyzer() -> NLPAnalyzer:
    """
    Return a singleton NLPAnalyzer instance.
    """
    if "nlp" not in _ai_instances:
        logger.info("Initializing NLPAnalyzer...")
        _ai_instances["nlp"] = NLPAnalyzer()
    return _ai_instances["nlp"]


def get_adversarial_detector() -> AdversarialDetector:
    """
    Return a singleton AdversarialDetector instance.
    """
    if "adversarial" not in _ai_instances:
        logger.info("Initializing AdversarialDetector...")
        _ai_instances["adversarial"] = AdversarialDetector()
    return _ai_instances["adversarial"]


def get_predictive_model() -> PredictiveThreatModel:
    """
    Return a singleton PredictiveThreatModel instance.
    """
    if "predictive" not in _ai_instances:
        logger.info("Initializing PredictiveThreatModel...")
        _ai_instances["predictive"] = PredictiveThreatModel()
    return _ai_instances["predictive"]
