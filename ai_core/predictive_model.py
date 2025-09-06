# -*- coding: utf-8 -*-
"""
Predictive Threat Model for Sentenial-X
---------------------------------------

Uses historical telemetry, AI insights, and statistical models to
predict potential threats or attacks before they occur.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from .config import MODEL_PATH_PREDICTIVE, PREDICTIVE_RISK_THRESHOLD, USE_GPU

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEVICE = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"


class PredictiveThreatModel:
    """
    AI-based predictive threat model.
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH_PREDICTIVE,
        risk_threshold: float = PREDICTIVE_RISK_THRESHOLD,
    ):
        self.model_path = model_path
        self.risk_threshold = risk_threshold
        self.device = DEVICE
        self.model: Optional[nn.Module] = None
        self._load_model()

    def _load_model(self) -> None:
        """
        Load predictive threat model weights.
        """
        try:
            logger.info("Loading predictive threat model from %s", self.model_path)
            self.model = torch.load(self.model_path, map_location=self.device)
            self.model.eval()
            logger.info("Predictive threat model loaded successfully on device: %s", self.device)
        except Exception as e:
            logger.exception("Failed to load predictive threat model")
            raise RuntimeError(f"Predictive model loading failed: {e}")

    def predict(self, features: List[float]) -> Dict[str, Any]:
        """
        Predict threat risk based on input features.

        Args:
            features: List of numerical features representing system state.

        Returns:
            dict with keys: 'risk_score', 'is_risky'
        """
        if not features or self.model is None:
            return {"risk_score": 0.0, "is_risky": False}

        try:
            x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                output = self.model(x)
                if isinstance(output, tuple) or isinstance(output, list):
                    output = output[0]  # support models returning tuples

                score = torch.sigmoid(output).item()  # ensure 0-1 probability

            is_risky = score >= self.risk_threshold

            return {"risk_score": float(score), "is_risky": bool(is_risky)}
        except Exception as e:
            logger.exception("Failed to predict threat risk")
            return {"risk_score": 0.0, "is_risky": False, "error": str(e)}
