"""
ai_core/utils.py
----------------
Utility functions for Sentenial-X AI Core.
Provides helpers for:
- Normalizing input data
- Evaluating model predictions
- Processing feedback for training
- Logging metrics
"""

import logging
from typing import Any, Dict, List
from api.utils.logger import init_logger
import numpy as np
# ai_core/utils.py
import logging

logger = logging.getLogger("SentenialX.AICore")
logger.setLevel(logging.INFO)

def preprocess_input(text: str) -> str:
    """
    Normalize and clean logs, prompts, or text inputs.
    """
    return " ".join(text.strip().split())

def log_info(message: str):
    logger.info(message)

def log_warn(message: str):
    logger.warning(message)
logger = init_logger("ai_core.utils")


# ----------------------------
# Data Normalization
# ----------------------------
def normalize_features(features: Dict[str, Any]) -> Dict[str, float]:
    """
    Normalize numeric features to [0, 1] range.
    Non-numeric fields are ignored.
    """
    normalized = {}
    for k, v in features.items():
        try:
            v = float(v)
            normalized[k] = min(max(v, 0.0), 1.0)  # clamp to [0,1]
        except (ValueError, TypeError):
            logger.debug("Skipping non-numeric feature: %s", k)
            continue
    return normalized


# ----------------------------
# Model Evaluation
# ----------------------------
def evaluate_predictions(predictions: List[Any], labels: List[Any]) -> Dict[str, float]:
    """
    Evaluate model predictions against true labels.
    Returns metrics like accuracy, precision, recall.
    """
    if not predictions or not labels or len(predictions) != len(labels):
        logger.warning("Invalid predictions/labels length")
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0}

    predictions = np.array(predictions)
    labels = np.array(labels)

    accuracy = float(np.sum(predictions == labels) / len(labels))
    precision = float(np.sum(predictions & labels) / (np.sum(predictions) + 1e-8))
    recall = float(np.sum(predictions & labels) / (np.sum(labels) + 1e-8))

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }
    logger.info("Evaluation metrics: %s", metrics)
    return metrics


# ----------------------------
# Feedback Processing
# ----------------------------
def process_feedback(feedback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Preprocess feedback data for training.
    Ensures required fields exist and normalizes numeric values.
    """
    processed = []
    for entry in feedback:
        threat = entry.get("threat", {})
        expected_severity = entry.get("expected_severity", "medium")
        features = normalize_features(threat)
        processed.append({
            "features": features,
            "expected_severity": expected_severity
        })
    logger.info("Processed %d feedback entries", len(processed))
    return processed


# ----------------------------
# Logging Helpers
# ----------------------------
def log_model_metrics(model_name: str, metrics: Dict[str, float]):
    """
    Log evaluation metrics for a given model.
    """
    logger.info("Metrics for model %s: %s", model_name, metrics)


# ----------------------------
# CLI / Test Example
# ----------------------------
if __name__ == "__main__":
    sample_feedback = [
        {"threat": {"cpu": 0.8, "memory": 0.6, "payload_size": 1024}, "expected_severity": "high"},
        {"threat": {"cpu": 0.1, "memory": 0.2}, "expected_severity": "low"},
    ]
    processed = process_feedback(sample_feedback)
    print("Processed feedback:", processed)

    metrics = evaluate_predictions([1, 0, 1], [1, 0, 0])
    print("Sample metrics:", metrics)
