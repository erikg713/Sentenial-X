# core/neural_engine/learning_engine.py

import os
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from core.neural_engine.deep_plugin import DeepLearningEngine
import os
import json
import logging
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LearningEngine")


class LearningEngine:
    """
    Handles training, prediction, and feedback for threat detection models.
    Supports both Random Forest (sklearn) and Deep Learning engines.
    """
    def __init__(self, mode: str = "rf", model_dir: str = "models", max_buffer: int = 2000):
        self.mode = mode.lower()
        self.model_dir = model_dir
        self.max_buffer = max_buffer

        self.rf_model_path = os.path.join(model_dir, "rf_model.pkl")
        self.deep_model_path = os.path.join(model_dir, "deep_model.pt")
        self.config_path = os.path.join(model_dir, "config.json")

        self.data_buffer = deque(maxlen=max_buffer)
        self.labeled_data = []
        self.lock = threading.Lock()
        self.is_training = False

        self.rf_model = None
        self.label_encoder = LabelEncoder()
        self.model_version = 0
        self.trained = False

        os.makedirs(model_dir, exist_ok=True)

        if self.mode == "rf":
            self._load_rf_model()
        elif self.mode == "deep":
            try:
                from core.neural_engine.deep_plugin import DeepLearningEngine
                self.deep_model = DeepLearningEngine(model_path=self.deep_model_path)
                self.deep_model.load_model()
            except ImportError:
                logger.warning("DeepLearningEngine not found. Deep mode unavailable.")
                self.deep_model = None
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _load_rf_model(self):
        """
        Loads the Random Forest model if it exists, otherwise creates a new one.
        """
        if os.path.exists(self.rf_model_path):
            import joblib
            with open(self.rf_model_path, "rb") as f:
                data = joblib.load(f)
                self.rf_model = data["model"]
                self.label_encoder.classes_ = data["classes"]
                self.trained = True
            logger.info(f"Random Forest model loaded from {self.rf_model_path}")
        else:
            self.rf_model = RandomForestClassifier(n_estimators=100)
            logger.info("Initialized new Random Forest model.")

    def extract_features(self, event: dict) -> np.ndarray:
        """
        Converts a raw event dict to a feature vector (numpy array).
        Adjust this method based on your actual event schema.
        """
        try:
            ip_octets = [int(octet) for octet in event.get('ip', '0.0.0.0').split('.')]
            process_hash = hash(event.get('process', ''))
            payload_hash = hash(event.get('payload_hash', ''))
            severity = event.get('severity', 0)
            timestamp = int(datetime.strptime(
                event.get('timestamp', '1970-01-01T00:00:00'), "%Y-%m-%dT%H:%M:%S"
            ).timestamp())
            features = np.array(ip_octets + [process_hash, payload_hash, severity, timestamp])
            return features
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None

    def ingest_event(self, event: dict):
        """
        Ingests a raw telemetry/threat event for future training.
        """
        features = self.extract_features(event)
        if features is not None:
            with self.lock:
                self.data_buffer.append(features)
            logger.debug("Event ingested into buffer.")

    def receive_feedback(self, features, label):
        """
        Receives analyst or auto-generated feedback for supervised retraining.
        """
        with self.lock:
            self.labeled_data.append((features, label))
        logger.debug("Feedback received.")

    def retrain_if_ready(self, min_samples: int = 20):
        """
        Retrains the model asynchronously if enough labeled data is available.
        """
        with self.lock:
            if self.is_training or len(self.labeled_data) < min_samples:
                return
            self.is_training = True
            training_data = self.labeled_data.copy()
            self.labeled_data.clear()

        t = threading.Thread(target=self._train_model, args=(training_data,))
        t.start()

    def _train_model(self, training_data):
        """
        Trains or retrains the model on the provided data.
        """
        logger.info(f"Training model on {len(training_data)} samples...")
        X = np.array([x for x, _ in training_data])
        y_labels = [str(lbl).lower() for _, lbl in training_data]
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

        with self.lock:
            self.model_version += 1
            self.is_training = False

        self.save_config()

    def select_action(self, features) -> str:
        """
        Predicts an action given feature input.
        """
        if self.mode == "rf":
            return self._rf_predict(features)
        elif self.mode == "deep" and self.deep_model:
            return self.deep_model.predict(features)
        else:
            logger.warning("No trained model available, defaulting to 'monitor'.")
            return "monitor"

    def _rf_predict(self, features):
        """
        Predicts using the Random Forest model and returns a mapped action.
        """
        if not self.trained or self.rf_model is None:
            return "monitor"
        try:
            pred = self.rf_model.predict([features])[0]
            label = self.label_encoder.inverse_transform([pred])[0]
            return {
                "benign": "monitor",
                "suspicious": "alert",
                "malicious": "isolate"
            }.get(label, "monitor")
        except Exception as e:
            logger.error(f"RF prediction error: {e}")
            return "monitor"

    def set_mode(self, mode: str):
        """
        Switches between 'rf' and 'deep' modes.
        """
        self.mode = mode.lower()
        if self.mode == "rf":
            self._load_rf_model()
        elif self.mode == "deep" and hasattr(self, 'deep_model') and self.deep_model:
            self.deep_model.load_model()
        else:
            logger.warning(f"Unknown or unavailable mode: {mode}")

        self.save_config()

    def save_config(self):
        """
        Saves model metadata/configuration.
        """
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
        """
        Loads model configuration if available.
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    config = json.load(f)
                self.set_mode(config.get("mode", "rf"))
                self.model_version = config.get("model_version", 0)
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
        else:
            logger.info("No config file found, using defaults.")


class BehavioralIDS:
    """
    Example Behavioral Intrusion Detection System using LearningEngine.
    """
    def __init__(self, mode="rf"):
        self.engine = LearningEngine(mode=mode)
        self.log = []

    def analyze(self, packet_features):
        """
        Analyzes packet features and responds accordingly.
        """
        decision = self.engine.select_action(packet_features)
        self._respond(decision, packet_features)

    def _respond(self, decision, features):
        """
        Takes action based on the decision.
        """
        if decision == "monitor":
            logger.info("[IDS] Normal activity.")
        elif decision == "alert":
            logger.warning("[IDS] Suspicious pattern detected!")
            self.log.append({"action": "alert", "features": features})
        elif decision == "isolate":
            logger.critical("[IDS] Malicious pattern! Isolating source.")
            self.log.append({"action": "isolate", "features": features})
            # Here, you would trigger firewall rules, etc.

class LearningEngine:
    def __init__(self, mode="deep", model_dir="models"):
        self.mode = mode.lower()  # 'deep' or 'rf'
        self.model_dir = model_dir
        self.rf_model_path = os.path.join(model_dir, "rf_model.pkl")
        self.deep_model = DeepLearningEngine(model_path=os.path.join(model_dir, "deep_model.pt"))
        self.rf_model = None
        self.label_encoder = LabelEncoder()
        self.trained = False

        os.makedirs(model_dir, exist_ok=True)
        if self.mode == "rf":
            self._load_rf_model()
        else:
            self.deep_model.load_model()

    def _load_rf_model(self):
        if os.path.exists(self.rf_model_path):
            with open(self.rf_model_path, "rb") as f:
                data = joblib.load(f)
                self.rf_model = data["model"]
                self.label_encoder.classes_ = data["classes"]
                self.trained = True
        else:
            self.rf_model = RandomForestClassifier(n_estimators=100)

    def train(self, data, labels):
        labels = [str(lbl).lower() for lbl in labels]
        X = np.array(data)
        y = self.label_encoder.fit_transform(labels)

        if self.mode == "rf":
            self.rf_model.fit(X, y)
            joblib.dump({
                "model": self.rf_model,
                "classes": self.label_encoder.classes_
            }, self.rf_model_path)
            self.trained = True
        else:
            self.deep_model.train(X, labels)
            self.trained = True

    def select_action(self, features):
        if self.mode == "rf":
            return self._rf_predict(features)
        else:
            return self.deep_model.predict(features)

    def _rf_predict(self, features):
        if not self.trained or self.rf_model is None:
            return "monitor"
        try:
            pred = self.rf_model.predict([features])[0]
            label = self.label_encoder.inverse_transform([pred])[0]
            return {
                "benign": "monitor",
                "suspicious": "alert",
                "malicious": "isolate"
            }.get(label, "monitor")
        except:
            return "monitor"

    def set_mode(self, mode):
        self.mode = mode.lower()
        if self.mode == "rf":
            self._load_rf_model()
        else:
            self.deep_model.load_model()

    def export_config(self, path="models/config.json"):
        with open(path, "w") as f:
            json.dump({"mode": self.mode}, f)

    def import_config(self, path="models/config.json"):
        if os.path.exists(path):
            with open(path, "r") as f:
                config = json.load(f)
                self.set_mode(config.get("mode", "rf"))from core.neural_engine.learning_engine import LearningEngine

class BehavioralIDS:
    def __init__(self):
        self.engine = LearningEngine(mode="deep")
        self.log = []

    def analyze(self, packet_features):
        decision = self.engine.select_action(packet_features)
        self._respond(decision, packet_features)

    def _respond(self, decision, features):
        if decision == "monitor":
            print("[IDS] Normal activity.")
        elif decision == "alert":
            print("[IDS] Suspicious pattern detected!")
            self.log.append({"action": "alert", "features": features})
        elif decision == "isolate":
            print("[IDS] Malicious pattern! Isolating source.")
            self.log.append({"action": "isolate", "features": features})
            # trigger firewall rule or system response# core/neural_engine/learning_engine.py
import threading
import time
import json
import os
from collections import deque
from datetime import datetime

class LearningEngine:
    def __init__(self, model=None, max_buffer_size=2000, model_save_path="models/learning_model.json"):
        self.model = model  # placeholder for AI/ML model object
        self.data_buffer = deque(maxlen=max_buffer_size)  # Recent raw feature vectors
        self.labeled_data = []  # (features, label) for supervised retraining
        self.lock = threading.Lock()
        self.is_training = False
        self.model_save_path = model_save_path
        self.model_version = 0
        self.load_model()

    def ingest_event(self, event):
        """
        Ingests raw telemetry or threat event.
        Extract features and add to buffer for later training/analysis.
        """
        features = self.extract_features(event)
        if features:
            with self.lock:
                self.data_buffer.append(features)

    def extract_features(self, event):
        """
        Convert raw event dict to feature vector.
        Customize per telemetry schema.
        """
        # Example: simplistic feature extraction
        # Assume event dict has keys: 'ip', 'process', 'payload_hash', 'severity', 'timestamp'
        try:
            features = {
                'ip_octets': [int(octet) for octet in event.get('ip', '0.0.0.0').split('.')],
                'process_hash': hash(event.get('process', '')),
                'payload_hash': hash(event.get('payload_hash', '')),
                'severity': event.get('severity', 0),
                'timestamp': int(datetime.strptime(event.get('timestamp', '1970-01-01T00:00:00'), "%Y-%m-%dT%H:%M:%S").timestamp())
            }
            return features
        except Exception as e:
            print(f"[LearningEngine] Feature extraction error: {e}")
            return None

    def receive_feedback(self, features, label):
        """
        Receive analyst or auto-generated feedback.
        Label is typically 'benign', 'malicious', or specific threat class.
        """
        with self.lock:
            self.labeled_data.append((features, label))

    def retrain_if_ready(self):
        """
        Periodically check if enough labeled data is collected.
        Retrain the model asynchronously.
        """
        with self.lock:
            if self.is_training or len(self.labeled_data) < 20:
                return
            self.is_training = True
            training_data = self.labeled_data.copy()
            self.labeled_data.clear()

        threading.Thread(target=self._train_model, args=(training_data,)).start()

    def _train_model(self, training_data):
        """
        Placeholder training logic.
        Integrate with your ML framework here.
        """
        print(f"[LearningEngine] Training model on {len(training_data)} samples...")
        time.sleep(5)  # simulate training time

        # Dummy "model update": increment version
        with self.lock:
            self.model_version += 1
            self.is_training = False

        self.save_model()
        print(f"[LearningEngine] Model retrained. Version: {self.model_version}")

    def select_action(self, threat_features):
        """
        Given threat features, select best adaptive countermeasure.
        Placeholder: returns string action.
        """
        # TODO: Use model.predict(threat_features) once integrated
        # Simple heuristic fallback:
        severity = threat_features.get('severity', 0)
        if severity > 7:
            return "isolate"
        elif severity > 4:
            return "quarantine"
        else:
            return "monitor"

    def save_model(self):
        """
        Save current model metadata (and serialized weights if any).
        """
        data = {
            "model_version": self.model_version,
            "timestamp": datetime.utcnow().isoformat(),
        }
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        with open(self.model_save_path, "w") as f:
            json.dump(data, f)

    def load_model(self):
        """
        Load model metadata (and weights if any).
        """
        if not os.path.exists(self.model_save_path):
            print("[LearningEngine] No saved model found. Starting fresh.")
            return
        try:
            with open(self.model_save_path, "r") as f:
                data = json.load(f)
            self.model_version = data.get("model_version", 0)
            print(f"[LearningEngine] Loaded model version {self.model_version}")
        except Exception as e:
            print(f"[LearningEngine] Failed to load model: {e}")
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import threading
import time

class LearningEngine:
    def __init__(self, model_path="models/rf_model.pkl"):
        self.model_path = model_path
        self.data_buffer = []
        self.labels = []
        self.label_encoder = LabelEncoder()
        self.model = None
        self.model_version = 0
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model, self.label_encoder, self.model_version = pickle.load(f)
            print(f"[LearningEngine] Loaded model version {self.model_version}")
        else:
            self.model = RandomForestClassifier(n_estimators=50)
            print("[LearningEngine] No saved model found. Starting fresh.")

    def save_model(self):
        with open(self.model_path, 'wb') as f:
            pickle.dump((self.model, self.label_encoder, self.model_version), f)
        print(f"[LearningEngine] Model saved. Version: {self.model_version}")

    def ingest_event(self, event):
        features = self.extract_features(event)
        self.data_buffer.append(features)
        # Label is unknown at this stage, so keep labels and data aligned later
        print(f"[LearningEngine] Event ingested with features: {features}")

    def extract_features(self, event):
        # Simple numeric feature vector from event dict
        # Customize your feature extraction here
        severity = event.get('severity', 0)
        # Hash to numeric feature: simple sum of ASCII codes (not ideal but demo)
        payload_hash = event.get('payload_hash', '')
        hash_val = sum(ord(c) for c in payload_hash) % 1000
        return [severity, hash_val]

    def receive_feedback(self, features, label):
        self.data_buffer.append(features)
        self.labels.append(label)
        print(f"[LearningEngine] Received feedback for features {features} with label '{label}'")

    def retrain_if_ready(self):
        if len(self.labels) < 5:
            print("[LearningEngine] Not enough labeled data to retrain.")
            return
        print(f"[LearningEngine] Training model on {len(self.labels)} samples...")
        # Encode string labels to numbers
        y_encoded = self.label_encoder.fit_transform(self.labels)
        X = np.array(self.data_buffer[-len(self.labels):])  # last n labeled samples only
        self.model.fit(X, y_encoded)
        self.model_version += 1
        self.save_model()
        print(f"[LearningEngine] Model retrained. Version: {self.model_version}")

    def select_action(self, features):
        if self.model is None or not self.labels:
            print("[LearningEngine] Model not ready, defaulting to monitor action.")
            return "monitor"
        X = np.array(features).reshape(1, -1)
        pred_encoded = self.model.predict(X)[0]
        pred_label = self.label_encoder.inverse_transform([pred_encoded])[0]
        # Map labels to actions (customize this)
        action_map = {
            "benign": "monitor",
            "malicious": "isolate",
            "suspicious": "alert"
        }
        action = action_map.get(pred_label, "monitor")
        return action

    def start_retrain_scheduler(self, interval_sec=300):
        def retrain_loop():
            while True:
                self.retrain_if_ready()
                time.sleep(interval_sec)
        t = threading.Thread(target=retrain_loop, daemon=True)
        t.start()
