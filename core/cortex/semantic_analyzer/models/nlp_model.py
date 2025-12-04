#!/usr/bin/env python3
"""
Sentenial-X :: Semantic Analyzer NLP Model
==========================================

Purpose:
    Lightweight NLP model specifically for semantic analyzer:
        - Converts event/log text to feature vectors
        - Performs threat intent classification
        - Pluggable for TF-IDF or small transformer embeddings
        - Compatible with SemanticAnalyzer and ZeroDayPredictor
"""

import os
import json
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# ------------------------------------------------------------
# NLP Model Wrapper for Semantic Analyzer
# ------------------------------------------------------------
class SemanticNLPModel:
    """
    Lightweight NLP model for semantic feature extraction
    """

    def __init__(self, backend: str = "tfidf", model_path: Optional[str] = None, embedding_dim: int = 128):
        """
        backend: "tfidf" | "transformer"
        model_path: optional pre-trained path
        embedding_dim: only relevant for transformer embeddings
        """
        self.backend = backend
        self.embedding_dim = embedding_dim

        if backend == "tfidf":
            self.vectorizer = TfidfVectorizer(max_features=5000)
            self.model = LogisticRegression(max_iter=500)
            if model_path and os.path.exists(model_path):
                self.load(model_path)

        elif backend == "transformer":
            if not HAS_TRANSFORMERS:
                raise ImportError("Transformers library not installed")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(model_path or "distilbert-base-uncased")
            self.model = AutoModel.from_pretrained(model_path or "distilbert-base-uncased")
            self.model.to(self.device)
            self.model.eval()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # --------------------------------------------------------
    # Fit / Train (TF-IDF backend)
    # --------------------------------------------------------
    def fit(self, texts: List[str], labels: List[int]):
        if self.backend != "tfidf":
            raise NotImplementedError("Fine-tuning only implemented for TF-IDF backend")
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)

    # --------------------------------------------------------
    # Predict probabilities
    # --------------------------------------------------------
    def predict_proba(self, texts: List[str]) -> List[Dict[str, float]]:
        if self.backend == "tfidf":
            X = self.vectorizer.transform(texts)
            probs = self.model.predict_proba(X)
            return [{str(i): float(p) for i, p in enumerate(prob)} for prob in probs]

        elif self.backend == "transformer":
            all_embeddings = []
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=128,
                    return_tensors="pt"
                )
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    # mean pooling
                    emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
                    all_embeddings.append(emb.tolist())
            return [{"embedding": emb} for emb in all_embeddings]

    # --------------------------------------------------------
    # Save / Load
    # --------------------------------------------------------
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        if self.backend == "tfidf":
            joblib.dump(self.vectorizer, os.path.join(path, "vectorizer.pkl"))
            joblib.dump(self.model, os.path.join(path, "model.pkl"))
        elif self.backend == "transformer":
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

    def load(self, path: str):
        if self.backend == "tfidf":
            self.vectorizer = joblib.load(os.path.join(path, "vectorizer.pkl"))
            self.model = joblib.load(os.path.join(path, "model.pkl"))
        elif self.backend == "transformer":
            if not HAS_TRANSFORMERS:
                raise ImportError("Transformers library not installed")
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModel.from_pretrained(path)
            self.model.to(self.device)
            self.model.eval()


# ------------------------------------------------------------
# CLI interface
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Semantic Analyzer NLP Model CLI")
    parser.add_argument("--texts", required=True, help="JSON array of texts to process")
    parser.add_argument("--backend", default="tfidf", choices=["tfidf", "transformer"], help="Model backend")
    parser.add_argument("--model_path", default=None, help="Load pretrained model from path")
    args = parser.parse_args()

    texts = json.loads(args.texts)
    nlp_model = SemanticNLPModel(backend=args.backend, model_path=args.model_path)
    outputs = nlp_model.predict_proba(texts)
    print(json.dumps(outputs, indent=4))
