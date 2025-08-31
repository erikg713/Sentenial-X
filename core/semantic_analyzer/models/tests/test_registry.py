# core/semantic_analyzer/models/tests/test_registry.py

import pytest
from core.semantic_analyzer.models.registry import ModelRegistry


class DummyModel:
    def __init__(self, name="dummy"):
        self.name = name

    def predict(self, x):
        return f"predicted-{x}"


def test_register_and_retrieve_model():
    registry = ModelRegistry()
    dummy_model = DummyModel("test")

    registry.register("dummy", dummy_model)
    retrieved = registry.get("dummy")

    assert retrieved is dummy_model
    assert isinstance(retrieved, DummyModel)
    assert retrieved.name == "test"


def test_register_duplicate_model_raises():
    registry = ModelRegistry()
    dummy_model = DummyModel()

    registry.register("duplicate", dummy_model)

    with pytest.raises(ValueError) as excinfo:
        registry.register("duplicate", dummy_model)

    assert "already registered" in str(excinfo.value)


def test_get_non_existent_model_raises():
    registry = ModelRegistry()

    with pytest.raises(KeyError) as excinfo:
        registry.get("not_registered")

    assert "not_registered" in str(excinfo.value)


def test_list_models():
    registry = ModelRegistry()
    registry.register("m1", DummyModel("m1"))
    registry.register("m2", DummyModel("m2"))

    models = registry.list_models()
    assert "m1" in models
    assert "m2" in models
    assert len(models) == 2


def test_clear_models():
    registry = ModelRegistry()
    registry.register("m1", DummyModel("m1"))
    registry.register("m2", DummyModel("m2"))

    assert len(registry.list_models()) == 2

    registry.clear()

    assert registry.list_models() == []
