# core/neural_engine/learning_engine.py
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
