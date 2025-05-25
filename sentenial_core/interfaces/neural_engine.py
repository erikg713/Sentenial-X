"""
sentenial_core.interfaces.neural_engine

Neural detection engine for Sentenial-X-A.I.
Supports incremental online training for continuous threat adaptation.

Author: Erik G. <dev713@github.com>
"""

from typing import Any, Dict, Optional
from datetime import datetime
import logging
import threading
import joblib
import os

from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
import numpy as np

logger = logging.getLogger("sentenial_core.neural")
logger.setLevel(logging.INFO)

MODEL_PATH = "sentenial_core/models/neural_engine.joblib"

class NeuralEngine:
    """
    Neural detection engine with online training.
    Designed for continuous learning and model updates.
    """
    def __init__(self):
        self.model = None
        self.classes_ = np.array(["benign", "threat"])
        self.lock = threading.Lock()
        self._load_or_init_model()

    def _load_or_init_model(self):
        # Load existing model or initialize a fresh one
        if os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                logger.info("Loaded existing neural engine model.")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self._init_model()
        else:
            self._init_model()

    def _init_model(self):
        # Use a simple SGDClassifier for online learning
        self.model = SGDClassifier(loss="log_loss", warm_start=True)
        # Initial partial_fit to set up classes
        dummy_X = np.zeros((1, 5))
        dummy_y = np.array(["benign"])
        self.model.partial_fit(dummy_X, dummy_y, classes=self.classes_)
        logger.info("Initialized new neural engine model.")

    def detect(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Example: extract features, run prediction
        features = self._extract_features(data)
        with self.lock:
            try:
                proba = self.model.predict_proba([features])[0]
                label = self.model.classes_[np.argmax(proba)]
                confidence = float(np.max(proba))
                logger.info(f"NeuralEngine detection: {label} (confidence={confidence:.2f})")
                return {
                    "threat_type": label,
                    "confidence": confidence,
                    "detected_at": datetime.utcnow().isoformat(),
                }
            except Exception as e:
                logger.error(f"Detection error: {e}")
                return None

    def update(self, data: Dict[str, Any], label: str):
        # Incrementally train on new data
        features = self._extract_features(data)
        with self.lock:
            try:
                self.model.partial_fit([features], [label])
                logger.info(f"Model updated with label={label}")
                self._save_model()
            except Exception as e:
                logger.error(f"Model update error: {e}")

    def retrain(self, X: np.ndarray, y: np.ndarray):
        # Full retrain on batch data
        X_shuffled, y_shuffled = shuffle(X, y, random_state=42)
        with self.lock:
            self.model = SGDClassifier(loss="log_loss", warm_start=True)
            self.model.partial_fit(X_shuffled, y_shuffled, classes=self.classes_)
            logger.info("Model retrained on full dataset.")
            self._save_model()

    def _extract_features(self, data: Dict[str, Any]) -> np.ndarray:
        # Example feature extraction (replace with your real logic)
        # For demo: five features, fill with 0 if missing
        keys = ["event", "count", "score", "role", "country"]
        features = []
        for k in keys:
            v = data.get(k)
            if isinstance(v, (int, float)):
                features.append(float(v))
            elif isinstance(v, str):
                features.append(float(hash(v) % 1000) / 1000)  # crude string hashing
            else:
                features.append(0.0)
        return np.array(features)

    def _save_model(self):
        try:
            joblib.dump(self.model, MODEL_PATH)
            logger.info("Neural engine model saved.")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
