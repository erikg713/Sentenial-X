"""
Sentenial X :: Self-Adaptive Engine - Anomaly Detection

This module provides an interface for detecting anomalies in telemetry streams.
It uses statistical thresholds and optionally machine learning models for behavior profiling.

Core Features:
- Z-score anomaly detection
- Model-based classification (e.g., RandomForest, IsolationForest)
- Rolling behavior analysis with windowing
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional
from sklearn.ensemble import IsolationForest

logger = logging.getLogger("SAEAnomaly")
logging.basicConfig(level=logging.INFO)


class SAEAnomalyDetector:
    def __init__(
        self,
        model: Optional[IsolationForest] = None,
        threshold: float = 2.5,
        use_model: bool = True
    ):
        """
        :param model: Optional pre-trained scikit-learn model
        :param threshold: Z-score threshold for statistical detection
        :param use_model: Toggle for using ML vs Z-score
        """
        self.model = model or IsolationForest(n_estimators=100, contamination=0.05)
        self.threshold = threshold
        self.use_model = use_model

    def fit(self, feature_vectors: List[List[float]]) -> None:
        """Train the model on historical normal data."""
        if not self.use_model:
            return
        logger.info("Fitting SAE model to normal baseline telemetry...")
        self.model.fit(feature_vectors)

    def detect(
        self,
        sample: List[float],
        history: List[List[float]] = []
    ) -> Dict[str, Any]:
        """
        Detect anomalies using either model-based or statistical approach.

        :param sample: Current telemetry feature vector
        :param history: Optional history of prior vectors (for Z-score)
        :return: Result with anomaly flag and confidence
        """
        if self.use_model:
            score = self.model.decision_function([sample])[0]
            is_anomaly = self.model.predict([sample])[0] == -1
            logger.debug(f"Model score: {score}")
            return {
                "anomaly": is_anomaly,
                "method": "model",
                "score": score
            }

        # Z-score based
        if not history:
            return {"anomaly": False, "method": "z-score", "reason": "no history"}

        arr = np.array(history + [sample])
        z_scores = np.abs((arr[-1] - arr[:-1].mean(axis=0)) / (arr[:-1].std(axis=0) + 1e-8))
        anomaly_dims = np.where(z_scores > self.threshold)[0]

        return {
            "anomaly": len(anomaly_dims) > 0,
            "method": "z-score",
            "anomalous_dimensions": anomaly_dims.tolist(),
            "z_scores": z_scores.tolist()
        }

