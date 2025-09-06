# -*- coding: utf-8 -*-
"""
Predictive Threat Model for Sentenial-X
---------------------------------------

Uses historical telemetry, AI insights, and statistical models to
predict potential threats or attacks before they occur.

Enhancements:
- Supports batch predictions
- Validates input features
- Logs analysis details
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

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
            if not isinstance(self.model, nn.Module):
                raise TypeError(f"Loaded object is not a PyTorch model: {type(self.model)}")
            self.model.eval()
            logger.info("Predictive threat model loaded successfully on device: %s", self.device)
        except Exception as e:
            logger.exception("Failed to load predictive threat model")
            raise RuntimeError(f"Predictive model loading failed: {e}")

    def predict(
        self, features: Union[List[float], List[List[float]]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Predict threat risk based on input features.

        Args:
            features: Single feature vector (List[float]) or batch (List of feature vectors).

        Returns:
            dict or list of dicts with keys: 'risk_score', 'is_risky'
        """
        if not features or self.model is None:
            return {"risk_score": 0.0, "is_risky": False}

        try:
            # Determine if single or batch
            if isinstance(features[0], list):
                x = torch.tensor(features, dtype=torch.float32, device=self.device)
            else:
                x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                output = self.model(x)
                if isinstance(output, (tuple, list)):
                    output = output[0]  # support models returning tuples

                scores = torch.sigmoid(output).flatten().tolist()

            results = []
            if isinstance(scores, float):
                results = [{"risk_score": scores, "is_risky": scores >= self.risk_threshold}]
            else:
                results = [{"risk_score": s, "is_risky": s >= self.risk_threshold} for s in scores]

            logger.debug("Predictive threat results: %s", results)
            return results if len(results) > 1 else results[0]

        except Exception as e:
            logger.exception("Failed to predict threat risk")
            return {"risk_score": 0.0, "is_risky": False, "error": str(e)}

    def validate_features(self, features: List[float]) -> bool:
        """
        Validate input features.
        Ensures numeric type and non-empty list.
        """
        if not isinstance(features, list) or not features:
            logger.warning("Invalid features: must be non-empty list")
            return False
        if not all(isinstance(f, (int, float)) for f in features):
            logger.warning("Invalid features: all elements must be int or float")
            return False
        return True
