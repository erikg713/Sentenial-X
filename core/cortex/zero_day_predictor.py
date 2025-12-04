#!/usr/bin/env python3
"""
Sentenial-X :: Zero Day Predictor Module
========================================

Purpose:
    Predicts probability of zero-day exploitation or anomaly-based threats
    by combining:
        - semantic signal analysis
        - behavioral anomaly metrics
        - statistical features
        - ML model predictions (pluggable)
        - confidence weighting

Architecture:
    ZeroDayPredictor → FeatureExtractor → ModelBackend → RiskScorer
"""

import argparse
import json
import numpy as np
from typing import Dict, Any, Optional


# ------------------------------------------------------------
# Feature Extraction
# ------------------------------------------------------------
class FeatureExtractor:
    """
    Converts raw signals, logs, or semantic outputs into numerical features.
    """

    def extract(self, payload: Dict[str, Any]) -> Dict[str, float]:
        """
        Input:
            payload = {
                "event_type": "...",
                "syscalls": [...],
                "entropy": float,
                "behavior_score": float,
                "anomaly_score": float,
                ...
            }

        Returns:
            dict of stable feature values
        """
        features = {}

        # Categorical → one-hot
        event_type = payload.get("event_type", "unknown").lower()
        features[f"event_type:{event_type}"] = 1.0

        # Syscall count
        syscalls = payload.get("syscalls", [])
        features["syscall_count"] = float(len(syscalls))

        # Entropy
        features["entropy"] = float(payload.get("entropy", 0.0))

        # Behavior score
        features["behavior_score"] = float(payload.get("behavior_score", 0.0))

        # Anomaly detection
        features["anomaly_score"] = float(payload.get("anomaly_score", 0.0))

        # Additional signals
        for k, v in payload.items():
            if isinstance(v, (int, float)) and k not in features:
                features[k] = float(v)

        return features


# ------------------------------------------------------------
# Model Backend
# ------------------------------------------------------------
class ModelBackend:
    """
    Abstracts ML model logic.
    Default: lightweight logistic-like scoring (no heavy deps).
    """

    def predict_proba(self, features: Dict[str, float]) -> float:
        """
        Returns a probability [0,1] representing the likelihood
        of zero-day behavior.

        This simple model:
            - weights high entropy
            - weights abnormal syscall patterns
            - weights anomaly score strongly
        """
        # Basic weighted sum
        entropy = features.get("entropy", 0.0)
        anomaly = features.get("anomaly_score", 0.0)
        syscount = features.get("syscall_count", 0.0)

        score = (0.55 * anomaly) + (0.30 * entropy) + (0.15 * min(syscount / 100, 1.0))

        # Sigmoid for probability
        return 1 / (1 + np.exp(-8 * (score - 0.5)))


# ------------------------------------------------------------
# Risk Scorer (fusion layer)
# ------------------------------------------------------------
class RiskScorer:
    def compute(self, model_prob: float, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Combines ML model probability + deterministic heuristics.
        """

        heuristic_boost = 0.0

        # High entropy binary → suspicious
        if features.get("entropy", 0) > 6.5:
            heuristic_boost += 0.10

        # Very high anomaly score amplifies risk
        if features.get("anomaly_score", 0) > 0.85:
            heuristic_boost += 0.15

        # Behavior score > critical threshold
        if features.get("behavior_score", 0) > 0.90:
            heuristic_boost += 0.20

        combined = min(model_prob + heuristic_boost, 1.0)

        return {
            "model_probability": round(model_prob, 5),
            "heuristic_boost": round(heuristic_boost, 5),
            "final_risk": round(combined, 5),
            "severity": self._label(combined)
        }

    @staticmethod
    def _label(risk: float) -> str:
        if risk < 0.25: return "Low"
        if risk < 0.50: return "Moderate"
        if risk < 0.75: return "High"
        return "Critical"


# ------------------------------------------------------------
# Main Predictor Interface
# ------------------------------------------------------------
class ZeroDayPredictor:
    def __init__(
        self,
        extractor: Optional[FeatureExtractor] = None,
        model: Optional[ModelBackend] = None,
        scorer: Optional[RiskScorer] = None,
    ):
        self.extractor = extractor or FeatureExtractor()
        self.model = model or ModelBackend()
        self.scorer = scorer or RiskScorer()

    def analyze(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        features = self.extractor.extract(payload)
        model_prob = self.model.predict_proba(features)
        return self.scorer.compute(model_prob, features)


# ------------------------------------------------------------
# CLI Interface
# ------------------------------------------------------------
def cli():
    parser = argparse.ArgumentParser(
        description="Sentenial-X Zero-Day Predictor CLI"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="JSON payload containing event features"
    )

    args = parser.parse_args()

    with open(args.input, "r") as f:
        payload = json.load(f)

    predictor = ZeroDayPredictor()
    result = predictor.analyze(payload)

    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    cli()
