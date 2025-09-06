# -*- coding: utf-8 -*-
"""
NLP Analyzer for Sentenial-X
----------------------------

Provides NLP-based threat analysis for logs, alerts, and text data.
Supports classification, entity extraction, and confidence scoring.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .config import MODEL_PATH_NLP, NLP_CONFIDENCE_THRESHOLD, USE_GPU

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEVICE = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"


class NLPAnalyzer:
    """
    NLP-based threat analyzer.
    """

    def __init__(self, model_path: str = MODEL_PATH_NLP, threshold: float = NLP_CONFIDENCE_THRESHOLD):
        self.model_path = model_path
        self.threshold = threshold
        self.device = DEVICE
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        """
        Load the pre-trained NLP model and tokenizer.
        """
        try:
            logger.info("Loading NLP model from %s", self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info("NLP model loaded successfully on device: %s", self.device)
        except Exception as e:
            logger.exception("Failed to load NLP model")
            raise RuntimeError(f"NLP model loading failed: {e}")

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze a text snippet for potential threats.

        Returns:
            dict with keys: 'label', 'score', 'is_threat'
        """
        if not text:
            return {"label": "unknown", "score": 0.0, "is_threat": False}

        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.softmax(outputs.logits, dim=-1)
                score, predicted_class = torch.max(scores, dim=-1)
                label = self.model.config.id2label[predicted_class.item()]

            is_threat = score.item() >= self.threshold

            return {
                "label": label,
                "score": float(score.item()),
                "is_threat": bool(is_threat)
            }
        except Exception as e:
            logger.exception("Failed to analyze text")
            return {"label": "error", "score": 0.0, "is_threat": False, "error": str(e)}
