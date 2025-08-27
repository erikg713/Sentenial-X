"""
Semantic Analyzer Models Package.

This package hosts all ML/NLP models, embeddings, and vectorizers 
used by the semantic analyzer core. Models can include:
- Transformer-based embeddings
- Custom classifiers
- Sequence labeling modules
- Vector similarity and clustering backends

Structure:
    embeddings/   - Pre-trained or fine-tuned embedding models
    classifiers/  - Custom classification models
    clustering/   - Semantic clustering utilities
    registry.py   - Central model registry for dynamic loading
"""

from importlib import import_module

# Expose a model registry for easy discovery and loading
from . import registry


def load_model(name: str):
    """
    Dynamically load a model by name from the registry.
    
    Args:
        name (str): Name of the model as registered in registry.py.
    
    Returns:
        object: Instantiated model class.
    """
    if name not in registry.MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in registry.")
    
    module_path, class_name = registry.MODEL_REGISTRY[name].rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, class_name)()
