"""
Sentenial-X AI Core: Model Loader
---------------------------------
Handles loading, initialization, and management of AI models
for WormGPT, Cortex, and other AI components.

Author: Sentenial-X Development Team
"""

import os
import logging
from typing import Any, Dict
from api.utils.logger import init_logger

logger = init_logger("ai_core.model_loader")


class ModelLoader:
    """
    Manages AI model loading and caching for production.
    """

    def __init__(self, model_dir: str = "models", default_model: str = "wormgpt"):
        self.model_dir = os.path.abspath(model_dir)
        self.default_model_name = default_model
        self.models: Dict[str, Any] = {}
        logger.info("ModelLoader initialized. Model directory: %s", self.model_dir)

    def load_model(self, model_name: str = None, force_reload: bool = False) -> Any:
        """
        Load a model from disk (or cache) by name.
        """
        model_name = model_name or self.default_model_name

        if model_name in self.models and not force_reload:
            logger.info("Returning cached model: %s", model_name)
            return self.models[model_name]

        model_path = os.path.join(self.model_dir, f"{model_name}.pt")  # PyTorch example
        if not os.path.exists(model_path):
            logger.error("Model file not found: %s", model_path)
            raise FileNotFoundError(f"Model '{model_name}' not found at {model_path}")

        # Placeholder for actual model loading (PyTorch, TensorFlow, HuggingFace, etc.)
        model = self._mock_load(model_path)
        self.models[model_name] = model
        logger.info("Model '%s' loaded successfully", model_name)
        return model

    def _mock_load(self, model_path: str) -> Dict[str, str]:
        """
        Mock loading function for demonstration.
        Replace with actual ML framework logic.
        """
        logger.debug("Mock loading model from %s", model_path)
        return {"model_path": model_path, "status": "loaded"}

    def list_models(self) -> Dict[str, Any]:
        """
        List all loaded models in memory.
        """
        return self.models

    def unload_model(self, model_name: str):
        """
        Remove a model from memory cache.
        """
        if model_name in self.models:
            del self.models[model_name]
            logger.info("Model '%s' unloaded from memory", model_name)
        else:
            logger.warning("Attempted to unload unknown model: %s", model_name)


# ------------------------
# Quick CLI Test
# ------------------------
if __name__ == "__main__":
    loader = ModelLoader()
    model = loader.load_model()
    print("Loaded model:", model)

    loader.load_model(force_reload=True)
    print("All loaded models:", loader.list_models())

    loader.unload_model("wormgpt")
    print("All loaded models after unload:", loader.list_models())
