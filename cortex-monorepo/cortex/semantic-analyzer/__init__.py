# -*- coding: utf-8 -*-
"""
Semantic Analyzer Package
-------------------------

Provides NLP and semantic analysis tools for Sentenial-X.
Used by Cortex and other modules to detect anomalies,
extract entities, and assess threat risk from textual input.
"""

from __future__ import annotations

from .models.base import BaseSemanticModel
from .models.text_analyzer import TextAnalyzer

__all__ = [
    "BaseSemanticModel",
    "TextAnalyzer",
]
