# -*- coding: utf-8 -*-
"""
TextAnalyzer for Sentenial-X
----------------------------

Implements NLP-based threat detection and semantic analysis.
"""

from __future__ import annotations
import logging
from typing import Any, Dict
import spacy

from .base import BaseSemanticModel

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class TextAnalyzer(BaseSemanticModel):
    """
    Analyze text for potential threats, anomalies, or sensitive content.
    Uses spaCy for entity recognition and basic semantic parsing.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        try:
            self.nlp = spacy.load(model_name)
            logger.info("TextAnalyzer loaded NLP model: %s", model_name)
        except Exception as e:
            logger.exception("Failed to load NLP model '%s'", model_name)
            raise RuntimeError(f"Failed to load NLP model '{model_name}': {e}")

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze input text and return structured semantic information.
        """
        if not text:
            return {"entities": [], "keywords": [], "score": 0.0, "is_threat": False}

        doc = self.nlp(text)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

        # Very simple keyword scoring for demo purposes
        keywords = [token.text for token in doc if token.is_alpha and token.is_stop is False]

        # Basic threat score: more entities/keywords increases risk
        score = min(1.0, 0.1 * len(entities) + 0.05 * len(keywords))
        is_threat = score > 0.5

        result = {
            "entities": entities,
            "keywords": keywords,
            "score": score,
            "is_threat": is_threat
        }

        logger.debug("TextAnalyzer result: %s", result)
        return result
