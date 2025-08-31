# core/semantic_analyzer/models/tests/test_models.py

import os
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from core.semantic_analyzer.models.base import BaseModel
from core.semantic_analyzer.models.cache import ModelCache
from core.semantic_analyzer.models.classifier import ClassifierModel
from core.semantic_analyzer.models.embedding import EmbeddingModel
from core.semantic_analyzer.models.onnx_runtime import ONNXRuntimeModel
from core.semantic_analyzer.models.registry import ModelRegistry
from core.semantic_analyzer.models.similarity import SimilarityModel
from core.semantic_analyzer.models.transformer import TransformerModel
from core.semantic_analyzer.models import utils


class DummyModel(BaseModel):
    """Simple test model for BaseModel behavior."""
    def _load_model(self):
        return "dummy_model"

    def predict(self, inputs):
        return [len(str(i)) for i in inputs]


def test_base_model_load_and_predict():
    model = DummyModel("dummy_path")
    assert model.model == "dummy_model"
    result = model.predict(["abc", "hello"])
    assert result == [3, 5]


def test_cache_store_and_retrieve():
    cache = ModelCache(max_size=2)
    cache.set("a", 123)
    cache.set("b", 456)
    assert cache.get("a") == 123
    cache.set("c", 789)  # should evict "b"
    assert cache.get("b") is None
    assert cache.get("c") == 789


def test_classifier_model_predict():
    clf = ClassifierModel("test_path", labels=["pos", "neg"])
    clf.model = MagicMock()
    clf.model.predict_proba = lambda x: np.array([[0.8, 0.2]])
    result = clf.predict(["test input"])
    assert isinstance(result, list)
    assert result[0]["label"] in ["pos", "neg"]
    assert "score" in result[0]


def test_embedding_model_vector_shape():
    emb = EmbeddingModel("embed_path", dim=16)
    emb.model = MagicMock()
    emb.model.encode = lambda x: np.ones((len(x), 16))
    vectors = emb.predict(["one", "two"])
    assert vectors.shape == (2, 16)


def test_onnx_runtime_model(tmp_path):
    fake_model_path = tmp_path / "fake.onnx"
    fake_model_path.write_text("not_a_real_model")

    with patch("onnxruntime.InferenceSession") as MockSession:
        mock_sess = MagicMock()
        mock_sess.get_inputs.return_value = [MagicMock(name="input")]
        mock_sess.run.return_value = [np.array([0.9])]
        MockSession.return_value = mock_sess

        model = ONNXRuntimeModel(str(fake_model_path))
        result = model.predict(np.array([[1, 2, 3]]))
        assert isinstance(result, list)


def test_registry_register_and_get():
    registry = ModelRegistry()
    dummy_model = DummyModel("dummy")
    registry.register("dummy", dummy_model)
    retrieved = registry.get("dummy")
    assert retrieved is dummy_model


def test_similarity_model():
    sim = SimilarityModel("sim_model")
    sim.model = MagicMock()
    sim.model.encode = lambda x: np.array([[1.0, 0.0], [0.0, 1.0]])
    result = sim.similarity(["a"], ["b"])
    assert result.shape == (1, 1)
    assert 0 <= result[0, 0] <= 1


def test_transformer_model_predict():
    transformer = TransformerModel("bert-base-uncased")
    transformer.model = MagicMock()
    transformer.tokenizer = MagicMock()
    transformer.tokenizer.return_value = {"input_ids": [1], "attention_mask": [1]}
    transformer.model.return_value = MagicMock()
    transformer.model.return_value.logits = np.array([[0.2, 0.8]])

    output = transformer.predict(["hello"])
    assert isinstance(output, np.ndarray)


def test_utils_normalize_and_cosine():
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([1, 2, 3])
    normed = utils.normalize(vec1)
    assert np.isclose(np.linalg.norm(normed), 1.0)

    similarity = utils.cosine_similarity(vec1, vec2)
    assert np.isclose(similarity, 1.0)


def test_utils_softmax_and_chunk():
    arr = np.array([1.0, 2.0, 3.0])
    sm = utils.softmax(arr)
    assert np.isclose(np.sum(sm), 1.0)

    chunks = list(utils.chunk_iterable([1, 2, 3, 4, 5], 2))
    assert chunks == [[1, 2], [3, 4], [5]]
