# core/semantic_analyzer/models/tests/test_embedding.py

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from core.semantic_analyzer.models.embedding import EmbeddingModel
from core.semantic_analyzer.models.registry import ModelRegistry


@pytest.fixture
def reset_registry():
    ModelRegistry.clear()


def test_transformer_embedding_mock(reset_registry):
    # Patch AutoModel and AutoTokenizer
    with patch("core.semantic_analyzer.models.embedding.AutoModel") as mock_model, \
         patch("core.semantic_analyzer.models.embedding.AutoTokenizer") as mock_tokenizer:

        dummy_tokenizer = MagicMock()
        dummy_tokenizer.return_value = {"input_ids": [1], "attention_mask": [1]}
        mock_tokenizer.from_pretrained.return_value = dummy_tokenizer

        dummy_model = MagicMock()
        dummy_model.config.hidden_size = 8
        dummy_model.return_value.last_hidden_state = np.random.rand(2, 1, 8)
        mock_model.from_pretrained.return_value = dummy_model

        em = EmbeddingModel(model_name="mock_transformer", backend="transformer")
        em.model = dummy_model
        em.tokenizer = dummy_tokenizer

        vectors = em.predict(["test1", "test2"])
        assert vectors.shape[0] == 2
        assert vectors.shape[1] == 8


def test_onnx_embedding_mock(reset_registry):
    dummy_session = MagicMock()
    dummy_session.run.return_value = [np.random.rand(2, 8)]
    with patch("core.semantic_analyzer.models.embedding.ort.InferenceSession") as mock_sess:
        mock_sess.return_value = dummy_session
        em = EmbeddingModel(model_name="mock_onnx", backend="onnx")
        em.session = dummy_session
        em.tokenizer = MagicMock()
        em.tokenizer.return_value = {"input_ids": np.array([[1]]), "attention_mask": np.array([[1]])}

        vectors = em.predict(["a", "b"])
        assert vectors.shape == (2, 8)
