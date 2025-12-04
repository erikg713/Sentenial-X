"""
Sentenial-X :: Cortex Models Package
===================================

Purpose:
    Expose all model classes in a unified namespace for Cortex components:
        - NLPModel (TF-IDF / BERT wrapper)
        - BertClassifier (HuggingFace BERT)
        - Other future ML/NLP models
"""

from .nlp_model import NLPModel
from .bert_classifier import BertClassifier

# Optional: default exports for easier imports
__all__ = [
    "NLPModel",
    "BertClassifier"
]
