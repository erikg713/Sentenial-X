# api/utils/risk_score.py

"""
Risk Scoring Utility
---------------------
This module provides flexible, weighted risk scoring logic for telemetry,
threat events, and anomaly detections. Scores can be used to prioritize
alerts, incidents, and automated responses.

Author: Sentenial-X Team
"""

import math
from typing import Dict, Any


class RiskScoreCalculator:
    """
    Risk Score Calculator
    ---------------------
    Provides weighted scoring based on severity, confidence, asset value,
    and other contextual factors.
    """

    DEFAULT_WEIGHTS = {
        "severity": 0.4,
        "confidence": 0.3,
        "asset_value": 0.2,
        "anomaly_score": 0.1,
    }

    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize the calculator with custom or default weights.

        Args:
            weights (dict): Optional custom weights for score factors.
        """
        self.weights = weights if weights else self.DEFAULT_WEIGHTS
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        """
        Normalize weights so they sum to 1.0.
        """
        total = sum(self.weights.values())
        if total == 0:
            raise ValueError("Risk score weights must sum to a non-zero value.")
        self.weights = {k: v / total for k, v in self.weights.items()}

    def calculate(self, factors: Dict[str, Any]) -> float:
        """
        Calculate a risk score based on input factors.

        Args:
            factors (dict): Dictionary containing factor values.
                Expected keys: severity, confidence, asset_value, anomaly_score.

        Returns:
            float: Normalized risk score between 0 and 100.
        """
        score = 0.0
        for factor, weight in self.weights.items():
            value = float(factors.get(factor, 0.0))
            normalized_value = max(0.0, min(value, 1.0))  # clamp between 0–1
            score += weight * normalized_value

        return round(score * 100, 2)

    def categorize(self, score: float) -> str:
        """
        Categorize the risk score into low/medium/high/critical.

        Args:
            score (float): Risk score (0–100).

        Returns:
            str: Risk category.
        """
        if score < 25:
            return "low"
        elif score < 50:
            return "medium"
        elif score < 75:
            return "high"
        return "critical"

    def calculate_with_category(self, factors: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate both score and category in one step.

        Args:
            factors (dict): Risk factor inputs.

        Returns:
            dict: Contains raw score and category.
        """
        score = self.calculate(factors)
        return {
            "score": score,
            "category": self.categorize(score),
        }


# Example usage
if __name__ == "__main__":
    calc = RiskScoreCalculator()
    example_factors = {
        "severity": 0.9,       # high severity
        "confidence": 0.8,     # strong detection confidence
        "asset_value": 0.7,    # important system
        "anomaly_score": 0.6,  # moderately anomalous
    }
    result = calc.calculate_with_category(example_factors)
    print("[Risk Score Example]", result)
