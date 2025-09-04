"""
Semantic Analyzer Models Package.

Hosts all ML/NLP models used in semantic analysis:
- Embeddings (transformer, word2vec, etc.)
- Classifiers (SVM, LogisticRegression, etc.)
- Clustering (semantic similarity, KMeans)
- Registry for dynamic model loading
"""

from importlib import import_module
from . import registry

def load_model(name: str):
    """
    Load a model dynamically by name from the registry.

    Args:
        name (str): Model name registered in registry.MODEL_REGISTRY.

    Returns:
        object: Instantiated model.
    """
    if name not in registry.MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in registry.")
    
    module_path, class_name = registry.MODEL_REGISTRY[name].rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, class_name)()
