# -*- coding: utf-8 -*-
"""
ai_core package initializer
---------------------------

Provides:
- Central access to NLPAnalyzer, AdversarialDetector, and PredictiveThreatModel
- Singleton-style lazy loading for all core AI components
- Clean import surface for Sentenial-X AI integration
"""

from __future__ import annotations

import logging
from typing import Optional

from .nlp_analyzer import NLPAnalyzer
from .adversarial_detector import AdversarialDetector
from .predictive_model import PredictiveThreatModel

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Singleton instances
_nlp_analyzer: Optional[NLPAnalyzer] = None
_adversarial_detector: Optional[AdversarialDetector] = None
_predictive_model: Optional[PredictiveThreatModel] = None


def get_nlp_analyzer() -> NLPAnalyzer:
    """Return a singleton NLPAnalyzer instance."""
    global _nlp_analyzer
    if _nlp_analyzer is None:
        _nlp_analyzer = NLPAnalyzer()
        logger.info("NLPAnalyzer initialized")
    return _nlp_analyzer


def get_adversarial_detector() -> AdversarialDetector:
    """Return a singleton AdversarialDetector instance."""
    global _adversarial_detector
    if _adversarial_detector is None:
        _adversarial_detector = AdversarialDetector()
        logger.info("AdversarialDetector initialized")
    return _adversarial_detector


def get_predictive_model() -> PredictiveThreatModel:
    """Return a singleton PredictiveThreatModel instance."""
    global _predictive_model
    if _predictive_model is None:
        _predictive_model = PredictiveThreatModel()
        logger.info("PredictiveThreatModel initialized")
    return _predictive_model


__all__ = [
    "NLPAnalyzer",
    "AdversarialDetector",
    "PredictiveThreatModel",
    "get_nlp_analyzer",
    "get_adversarial_detector",
    "get_predictive_model",
]
