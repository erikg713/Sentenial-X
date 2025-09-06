# -*- coding: utf-8 -*-
"""
AnomalyDetector for Sentenial-X
-------------------------------

Detects unusual patterns, deviations, or potential threats
in text logs, telemetry sequences, or structured event data.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List

import numpy as np
from sklearn.ensemble import IsolationForest

from .base import BaseSemanticModel

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class AnomalyDetector(BaseSemanticModel):
    """
    Detect anomalies in structured or numeric representations
    of events or textual embeddings.
    """

    def __init__(self, contamination: float = 0.01, random_state: int = 42):
        """
        Args:
            contamination: Estimated fraction of outliers in the data
            random_state: Random seed for reproducibility
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False
        logger.info("AnomalyDetector initialized with contamination=%s", contamination)

    def fit(self, X: List[List[float]]) -> None:
        """
        Fit the anomaly detection model on historical data.
        """
        if not X:
            logger.warning("No data provided for fitting AnomalyDetector")
            return
        X_np = np.array(X, dtype=float)
        self.model.fit(X_np)
        self.is_fitted = True
        logger.info("AnomalyDetector fitted on %d samples", X_np.shape[0])

    def analyze(self, X: List[float]) -> Dict[str, Any]:
        """
        Predict if the given input is anomalous.
        """
        if not self.is_fitted:
            logger.warning("AnomalyDetector not fitted. Returning default safe response.")
            return {"score": 0.0, "is_anomaly": False}

        X_np = np.array(X, dtype=float).reshape(1, -1)
        pred = self.model.predict(X_np)[0]  # 1 for normal, -1 for anomaly
        score = 1.0 if pred == -1 else 0.0

        result = {
            "score": score,
            "is_anomaly": bool(score)
        }

        logger.debug("AnomalyDetector result: %s", result)
        return result
