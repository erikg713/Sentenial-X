import yaml
import joblib
import os
from typing import Dict, Any
from .pipeline import preprocess


class ThreatClassifier:
    def __init__(self, model_path: str, rules_path: str):
        self.rules = self._load_rules(rules_path)
        self.model = self._load_model(model_path)

    def _load_rules(self, path: str):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _load_model(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}")
        return joblib.load(path)

    def classify(self, payload: str) -> Dict[str, Any]:
        """
        Hybrid classification using rules + ML model
        """
        # Rule-based detection
        for rule in self.rules.get("signatures", []):
            if rule["pattern"].lower() in payload.lower():
                return {"threat": True, "method": "rule", "label": rule["label"]}

        # ML-based detection
        features = preprocess(payload)
        prediction = self.model.predict([features])[0]
        proba = (
            self.model.predict_proba([features])[0][1]
            if hasattr(self.model, "predict_proba")
            else None
        )

        return {
            "threat": bool(prediction),
            "method": "ml",
            "label": "malicious" if prediction else "benign",
            "confidence": float(proba) if proba is not None else None,
        }
