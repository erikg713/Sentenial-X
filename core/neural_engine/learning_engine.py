"""
core/neural_engine/learning_engine.py

SentenialX Learning Engine — Handles adaptive training, prediction, and model management for cyber threat detection.
Supports Random Forest (scikit-learn) and deep learning backends.
"""

import os
import json
import logging
import threading
from collections import deque
from datetime import datetime
from typing import Optional, Any, Dict, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

try:
    from core.neural_engine.deep_plugin import DeepLearningEngine
except ImportError:
    DeepLearningEngine = None

# --- Logging Config ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s"
)
logger = logging.getLogger("LearningEngine")

# --- Feature Engineering Utilities ---

def extract_features(event: Dict[str, Any]) -> np.ndarray:
    """
    Convert a raw event dict into a numeric feature vector.
    Extend this logic as new features become relevant.
    """
    try:
        ip = event.get('ip', '0.0.0.0')
        ip_octets = [int(octet) for octet in ip.split('.')]
        process_hash = hash(event.get('process', ''))
        payload_hash = hash(event.get('payload_hash', ''))
        severity = int(event.get('severity', 0))
        timestamp = int(datetime.strptime(
            event.get('timestamp', '1970-01-01T00:00:00'),
            "%Y-%m-%dT%H:%M:%S"
        ).timestamp())
        return np.array(ip_octets + [process_hash, payload_hash, severity, timestamp], dtype=np.int64)
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return None

# --- Learning Engine ---

class LearningEngine:
    """
    Adaptive learning engine for threat detection.
    Handles model training, prediction, feedback, and versioning.
    Supports 'rf' (Random Forest) and 'deep' (DeepLearningEngine) backends.
    Thread-safe and production-ready.
    """
    def __init__(
        self,
        mode: str = "rf",
        model_dir: str = "models",
        max_buffer: int = 2000,
        min_retrain_samples: int = 20,
    ):
        self.mode = mode.lower()
        self.model_dir = model_dir
        self.max_buffer = max_buffer
        self.min_retrain_samples = min_retrain_samples

        self.rf_model_path = os.path.join(model_dir, "rf_model.pkl")
        self.deep_model_path = os.path.join(model_dir, "deep_model.pt")
        self.config_path = os.path.join(model_dir, "config.json")

        self.data_buffer = deque(maxlen=max_buffer)
        self.labeled_data: List[tuple] = []
        self.lock = threading.Lock()
        self.is_training = False
        self.trained = False
        self.model_version = 0

        self.rf_model: Optional[RandomForestClassifier] = None
        self.label_encoder = LabelEncoder()
        self.deep_model = None

        os.makedirs(self.model_dir, exist_ok=True)
        self._init_backend()
        self.load_config()

    def _init_backend(self):
        if self.mode == "rf":
            self._load_rf_model()
        elif self.mode == "deep":
            if DeepLearningEngine is not None:
                self.deep_model = DeepLearningEngine(model_path=self.deep_model_path)
                self.deep_model.load_model()
                self.trained = self.deep_model.is_trained if hasattr(self.deep_model, "is_trained") else True
            else:
                logger.warning("DeepLearningEngine unavailable; defaulting to Random Forest.")
                self.mode = "rf"
                self._load_rf_model()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _load_rf_model(self):
        if os.path.exists(self.rf_model_path):
            import joblib
            try:
                with open(self.rf_model_path, "rb") as f:
                    data = joblib.load(f)
                    self.rf_model = data["model"]
                    self.label_encoder.classes_ = data["classes"]
                    self.trained = True
                logger.info(f"Loaded Random Forest model from {self.rf_model_path}")
            except Exception as e:
                logger.error(f"Failed to load Random Forest model: {e}")
                self._init_new_rf()
        else:
            self._init_new_rf()

    def _init_new_rf(self):
        self.rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        self.trained = False
        logger.info("Initialized new Random Forest model.")

    def ingest_event(self, event: Dict[str, Any]):
        features = extract_features(event)
        if features is not None:
            with self.lock:
                self.data_buffer.append(features)
            logger.debug("Event ingested into buffer.")

    def receive_feedback(self, features: np.ndarray, label: str):
        with self.lock:
            self.labeled_data.append((features, str(label).lower()))
        logger.debug("Feedback received.")

    def retrain_if_ready(self):
        with self.lock:
            if self.is_training or len(self.labeled_data) < self.min_retrain_samples:
                return
            self.is_training = True
            training_data = self.labeled_data.copy()
            self.labeled_data.clear()
        threading.Thread(target=self._train_model, args=(training_data,), daemon=True).start()

    def _train_model(self, training_data: List[tuple]):
        logger.info(f"Training model on {len(training_data)} samples...")
        try:
            X = np.array([x for x, _ in training_data])
            y_labels = [lbl for _, lbl in training_data]
            y = self.label_encoder.fit_transform(y_labels)
            if self.mode == "rf":
                self.rf_model.fit(X, y)
                import joblib
                joblib.dump({
                    "model": self.rf_model,
                    "classes": self.label_encoder.classes_
                }, self.rf_model_path)
                self.trained = True
                logger.info("Random Forest model retrained and saved.")
            elif self.mode == "deep" and self.deep_model:
                self.deep_model.train(X, y_labels)
                self.trained = True
                logger.info("Deep Learning model retrained.")
            else:
                logger.warning("No valid model to train.")
        except Exception as e:
            logger.error(f"Training failed: {e}")
        finally:
            with self.lock:
                self.model_version += 1
                self.is_training = False
            self.save_config()

    def select_action(self, features: Any) -> str:
        if self.mode == "rf":
            return self._rf_predict(features)
        elif self.mode == "deep" and self.deep_model:
            return self.deep_model.predict(features)
        else:
            logger.warning("No trained model available, defaulting to 'monitor'.")
            return "monitor"

    def _rf_predict(self, features: Any) -> str:
        if not self.trained or self.rf_model is None:
            return "monitor"
        try:
            pred = self.rf_model.predict([features])[0]
            label = self.label_encoder.inverse_transform([pred])[0]
            action_map = {
                "benign": "monitor",
                "suspicious": "alert",
                "malicious": "isolate"
            }
            return action_map.get(label, "monitor")
        except Exception as e:
            logger.error(f"RF prediction error: {e}")
            return "monitor"

    def set_mode(self, mode: str):
        mode = mode.lower()
        if mode == self.mode:
            return
        self.mode = mode
        self._init_backend()
        self.save_config()

    def save_config(self):
        config = {
            "mode": self.mode,
            "model_version": self.model_version,
            "last_updated": datetime.utcnow().isoformat()
        }
        try:
            with open(self.config_path, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Config saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def load_config(self):
        if not os.path.exists(self.config_path):
            logger.info("No config file found, using defaults.")
            return
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
            self.set_mode(config.get("mode", "rf"))
            self.model_version = config.get("model_version", 0)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")

# --- Example Usage: Behavioral IDS ---

class BehavioralIDS:
    """
    Example Behavioral Intrusion Detection System using LearningEngine.
    """
    def __init__(self, mode: str = "rf"):
        self.engine = LearningEngine(mode=mode)
        self.log: List[Dict[str, Any]] = []

    def analyze(self, packet_features: Any):
        decision = self.engine.select_action(packet_features)
        self._respond(decision, packet_features)

    def _respond(self, decision: str, features: Any):
        if decision == "monitor":
            logger.info("[IDS] Normal activity.")
        elif decision == "alert":
            logger.warning("[IDS] Suspicious pattern detected!")
            self.log.append({"action": "alert", "features": features})
        elif decision == "isolate":
            logger.critical("[IDS] Malicious pattern! Isolating source.")
            self.log.append({"action": "isolate", "features": features})
            # Implement firewall or isolation logic here
