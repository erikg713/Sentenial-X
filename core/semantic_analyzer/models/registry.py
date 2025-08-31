# core/semantic_analyzer/models/registry.py

"""
Model Registry for Semantic Analyzer
------------------------------------

Provides a centralized registry to dynamically manage ML/NLP models such as
embeddings, classifiers, ONNX inference models, and caching layers.

Supports:
- Dynamic registration of models
- Lazy instantiation (load on first use)
- Reload / hot-swap capability
- Unified access across the system
"""

from typing import Dict, Type, Optional, Any, Callable
import threading


class ModelRegistry:
    """Thread-safe registry for models used in the semantic analyzer."""

    _lock = threading.Lock()
    _registry: Dict[str, Callable[[], Any]] = {}
    _instances: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[[], Any], overwrite: bool = False) -> None:
        """
        Register a model factory by name.

        Args:
            name (str): The unique identifier for the model.
            factory (Callable): A callable that returns an instance of the model.
            overwrite (bool): Whether to overwrite if already registered.
        """
        with cls._lock:
            if not overwrite and name in cls._registry:
                raise ValueError(f"Model '{name}' is already registered.")
            cls._registry[name] = factory
            if name in cls._instances:
                # If model exists, clear cached instance for reload
                del cls._instances[name]

    @classmethod
    def get(cls, name: str) -> Any:
        """
        Get an instance of a registered model. Lazy-loads if not instantiated yet.

        Args:
            name (str): The model name.

        Returns:
            Any: The model instance.
        """
        with cls._lock:
            if name not in cls._registry:
                raise KeyError(f"Model '{name}' is not registered.")
            if name not in cls._instances:
                cls._instances[name] = cls._registry[name]()
            return cls._instances[name]

    @classmethod
    def reload(cls, name: str) -> Any:
        """
        Force reload a model by re-invoking its factory.

        Args:
            name (str): The model name.

        Returns:
            Any: The new model instance.
        """
        with cls._lock:
            if name not in cls._registry:
                raise KeyError(f"Model '{name}' is not registered.")
            cls._instances[name] = cls._registry[name]()
            return cls._instances[name]

    @classmethod
    def list_models(cls) -> Dict[str, bool]:
        """
        List all registered models and whether they are currently instantiated.

        Returns:
            Dict[str, bool]: Mapping of model names -> instantiation status.
        """
        with cls._lock:
            return {name: (name in cls._instances) for name in cls._registry.keys()}

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister a model and remove its cached instance.

        Args:
            name (str): The model name.
        """
        with cls._lock:
            cls._registry.pop(name, None)
            cls._instances.pop(name, None)


# Example usage (could be moved into __init__.py of models/):
if __name__ == "__main__":
    # Example dummy model
    class DummyModel:
        def __init__(self):
            self.value = "I am alive"

    # Register dummy
    ModelRegistry.register("dummy", lambda: DummyModel())

    # Retrieve
    model = ModelRegistry.get("dummy")
    print(model.value)

    # Reload
    reloaded = ModelRegistry.reload("dummy")
    print(reloaded.value)

    # List
    print(ModelRegistry.list_models())
