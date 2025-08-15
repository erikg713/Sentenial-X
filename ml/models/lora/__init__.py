# sentenialx/models/lora/__init__.py

from pathlib import Path
from sentenialx.models.utils import load_torch_model, get_device
from sentenialx.models.artifacts import get_artifact_path, verify_artifact

LORA_MODEL_TYPE = "lora"  # Registry key for artifacts


def load_lora_weights(version: str = None, map_location=None):
    """
    Load only the LoRA adapter weights from artifacts.

    Args:
        version (str): Optional specific version to load.
        map_location: Optional torch device override.
    """
    if not verify_artifact(LORA_MODEL_TYPE):
        raise FileNotFoundError("LoRA weights missing or corrupted in artifacts registry.")

    weights_path = get_artifact_path(LORA_MODEL_TYPE)
    return load_torch_model(weights_path, map_location or get_device())


# Dev-only helper
def get_local_lora_path(filename="weights.bin"):
    """
    Returns the direct path to the LoRA weights file in the package (dev only).
    """
    return Path(__file__).resolve().parent / filename


# Import the public LoRA loader
from .lora_loader import load_lora_model

__all__ = ["load_lora_model", "load_lora_weights", "get_local_lora_path"]
