# core/semantic_analyzer/models/classifier.py

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from core.semantic_analyzer.models.cache import ModelCache

logger = logging.getLogger(__name__)


class SemanticClassifier:
    """
    A semantic text classifier that uses machine learning models
    (default: Logistic Regression + TF-IDF).
    Supports model persistence, caching, and probability-based predictions.
    """

    def __init__(
        self,
        model_path: str = "models/semantic_classifier.pkl",
        vectorizer_path: str = "models/vectorizer.pkl",
        labels_path: str = "models/labels.json",
    ):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.labels_path = labels_path

        self.model: Optional[LogisticRegression] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.labels: List[str] = []

        # In-memory cache
        self.cache = ModelCache(max_size=1000)

        self._load_artifacts()

    def _load_artifacts(self):
        """Load model, vectorizer, and labels if available."""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info(f"Loaded classifier model from {self.model_path}")
            if os.path.exists(self.vectorizer_path):
                self.vectorizer = joblib.load(self.vectorizer_path)
                logger.info(f"Loaded vectorizer from {self.vectorizer_path}")
            if os.path.exists(self.labels_path):
                with open(self.labels_path, "r", encoding="utf-8") as f:
                    self.labels = json.load(f)
                logger.info(f"Loaded labels: {self.labels}")
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")

    def train(self, texts: List[str], labels: List[str]) -> None:
        """
        Train a new classifier model.
        """
        logger.info("Training semantic classifier...")

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            max_features=5000
        )
        X = self.vectorizer.fit_transform(texts)

        self.labels = sorted(list(set(labels)))
        y = np.array([self.labels.index(label) for label in labels])

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, y)

        # Save artifacts
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)
        with open(self.labels_path, "w", encoding="utf-8") as f:
            json.dump(self.labels, f, indent=2)

        logger.info("Classifier training completed and artifacts saved.")

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict the most likely label for input text with confidence.
        """
        if not self.model or not self.vectorizer:
            raise ValueError("Classifier model is not trained or loaded.")

        if text in self.cache:
            return self.cache.get(text)

        X = self.vectorizer.transform([text])
        probs = self.model.predict_proba(X)[0]
        max_idx = np.argmax(probs)
        prediction = {
            "label": self.labels[max_idx],
            "confidence": float(probs[max_idx]),
            "probabilities": {
                self.labels[i]: float(probs[i]) for i in range(len(self.labels))
            }
        }

        self.cache.set(text, prediction)
        return prediction

    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict labels for multiple texts."""
        return [self.predict(t) for t in texts]

    def explain(self, text: str) -> Dict[str, float]:
        """
        Explain which features contribute most to classification.
        Uses model coefficients (only works for linear models).
        """
        if not self.model or not self.vectorizer:
            raise ValueError("Classifier model is not trained or loaded.")

        X = self.vectorizer.transform([text])
        feature_names = np.array(self.vectorizer.get_feature_names_out())

        if hasattr(self.model, "coef_"):
            coefs = self.model.coef_[0]
            contributions = X.toarray()[0] * coefs
            top_indices = np.argsort(contributions)[::-1][:10]
            return {
                feature_names[i]: float(contributions[i])
                for i in top_indices if contributions[i] != 0
            }
        else:
            return {"explanation": "Model does not support explanation."}
