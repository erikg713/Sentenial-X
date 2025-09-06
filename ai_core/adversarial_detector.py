# -*- coding: utf-8 -*-
"""
Adversarial Detector for Sentenial-X
------------------------------------

Detects adversarial or malicious inputs targeting AI systems,
including prompt injection, malicious payloads, or unusual patterns.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .config import MODEL_PATH_ADVERSARIAL, ADVERSARIAL_CONFIDENCE_THRESHOLD, USE_GPU

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEVICE = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"


class AdversarialDetector:
    """
    Detects adversarial or suspicious AI inputs.
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH_ADVERSARIAL,
        threshold: float = ADVERSARIAL_CONFIDENCE_THRESHOLD,
    ):
        self.model_path = model_path
        self.threshold = threshold
        self.device = DEVICE
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        """
        Load the adversarial detection model.
        """
        try:
            logger.info("Loading adversarial detection model from %s", self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Adversarial detection model loaded successfully on device: %s", self.device)
        except Exception as e:
            logger.exception("Failed to load adversarial detection model")
            raise RuntimeError(f"Adversarial model loading failed: {e}")

    def detect(self, text: str) -> Dict[str, Any]:
        """
        Analyze a text input for adversarial characteristics.

        Returns:
            dict with keys: 'label', 'score', 'is_adversarial'
        """
        if not text:
            return {"label": "unknown", "score": 0.0, "is_adversarial": False}

        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.softmax(outputs.logits, dim=-1)
                score, predicted_class = torch.max(scores, dim=-1)
                label = self.model.config.id2label[predicted_class.item()]

            is_adversarial = score.item() >= self.threshold

            return {
                "label": label,
                "score": float(score.item()),
                "is_adversarial": bool(is_adversarial),
            }
        except Exception as e:
            logger.exception("Failed to detect adversarial input")
            return {"label": "error", "score": 0.0, "is_adversarial": False, "error": str(e)}
