# core/semantic_analyzer/models/tests/__init__.py

"""
Unit tests for Sentenial-X semantic analyzer models.
Includes:
- Embedding models (Transformers & ONNX)
- Classifier models
- Transformer wrapper
- Similarity models
- ONNX runtime models

All production modules are tested using mocks to avoid
loading large models during CI/testing.
"""
