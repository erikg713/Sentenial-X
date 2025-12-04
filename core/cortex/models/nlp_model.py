#!/usr/bin/env python3
"""
Sentenial-X :: NLP Model Wrapper
================================

Purpose:
    Unified NLP model interface for Cortex:
        - Threat intent classification
        - Semantic embeddings
        - Text vectorization for downstream predictive models
        - Pluggable backend (BERT, TF-IDF, or other)
"""

import os
import json
from typing import List, Optional, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# ------------------------------------------------------------
# NLP Model Wrapper
# ------------------------------------------------------------
class NLPModel:
    """
    Pluggable NLP model interface
    """

    def __init__(self, backend: str = "tfidf", model_path: Optional[str] = None, num_labels: int = 2):
        """
        backend: "tfidf" | "bert"
        model_path: optional pre-trained model path
        """
        self.backend = backend
        self.num_labels = num_labels

        if backend == "tfidf":
            self.vectorizer = TfidfVectorizer(max_features=5000)
            self.model = LogisticRegression(max_iter=500)
            if model_path and os.path.exists(model_path):
                self.load(model_path)

        elif backend == "bert":
            if not HAS_TRANSFORMERS:
                raise ImportError("Transformers library not installed")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(model_path or "bert-base-uncased")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path or "bert-base-uncased",
                num_labels=num_labels
            )
            self.model.to(self.device)
            self.model.eval()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # --------------------------------------------------------
    # Fit / Train (for TF-IDF / classical models)
    # --------------------------------------------------------
    def fit(self, texts: List[str], labels: List[int]):
        if self.backend == "tfidf":
            X = self.vectorizer.fit_transform(texts)
            self.model.fit(X, labels)
        else:
            raise NotImplementedError("Fine-tuning not implemented in this wrapper. Use transformers directly.")

    # --------------------------------------------------------
    # Predict probability
    # --------------------------------------------------------
    def predict_proba(self, texts: List[str]) -> List[Dict[str, float]]:
        if self.backend == "tfidf":
            X = self.vectorizer.transform(texts)
            probs = self.model.predict_proba(X)
            return [{str(i): float(p) for i, p in enumerate(prob)} for prob in probs]

        elif self.backend == "bert":
            all_probs = []
            for text in texts:
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=128,
                    return_tensors="pt"
                )
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)

                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    prob = torch.softmax(outputs.logits, dim=1).squeeze(0)
                    all_probs.append({str(i): float(prob[i]) for i in range(prob.size(0))})
            return all_probs

    # --------------------------------------------------------
    # Save / Load
    # --------------------------------------------------------
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        if self.backend == "tfidf":
            joblib.dump(self.vectorizer, os.path.join(path, "vectorizer.pkl"))
            joblib.dump(self.model, os.path.join(path, "model.pkl"))
        elif self.backend == "bert":
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

    def load(self, path: str):
        if self.backend == "tfidf":
            self.vectorizer = joblib.load(os.path.join(path, "vectorizer.pkl"))
            self.model = joblib.load(os.path.join(path, "model.pkl"))
        elif self.backend == "bert":
            if not HAS_TRANSFORMERS:
                raise ImportError("Transformers library not installed")
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSequenceClassification.from_pretrained(path)
            self.model.to(self.device)
            self.model.eval()


# ------------------------------------------------------------
# CLI Interface
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NLPModel CLI")
    parser.add_argument("--texts", required=True, help="JSON array of texts to classify")
    parser.add_argument("--backend", default="tfidf", choices=["tfidf", "bert"], help="Model backend")
    parser.add_argument("--model_path", default=None, help="Path to load model from")
    args = parser.parse_args()

    texts = json.loads(args.texts)
    nlp_model = NLPModel(backend=args.backend, model_path=args.model_path)
    predictions = nlp_model.predict_proba(texts)
    print(json.dumps(predictions, indent=4))
