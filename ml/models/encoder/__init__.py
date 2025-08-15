# sentenialx/models/encoder/__init__.py

from pathlib import Path
from sentenialx.models.utils import load_torch_model, get_device
from sentenialx.models.artifacts import get_artifact_path, verify_artifact

ENCODER_MODEL_TYPE = "encoder"  # Registry key for artifacts


def load_encoder_model(version: str = None, map_location=None):
    """
    Load the text encoder model from artifacts.

    Args:
        version (str): Optional specific version to load.
        map_location: Optional torch device override.
    """
    if not verify_artifact(ENCODER_MODEL_TYPE):
        raise FileNotFoundError("Encoder model missing or corrupted in artifacts registry.")

    model_path = get_artifact_path(ENCODER_MODEL_TYPE)
    return load_torch_model(model_path, map_location or get_device())


# Dev-only helper
def get_local_encoder_path(filename="text_encoder.pt"):
    """
    Returns the direct path to the encoder model in the package (dev only).
    """
    return Path(__file__).resolve().parent / filename


# Import the main encoder class for convenience
from .text_encoder import ThreatTextEncoder

__all__ = ["ThreatTextEncoder", "load_encoder_model", "get_local_encoder_path"]
