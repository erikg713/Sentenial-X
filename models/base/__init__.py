# sentenialx/models/base/__init__.py

from pathlib import Path
from sentenialx.models.utils import load_torch_model, get_device
from sentenialx.models.artifacts import get_artifact_path, verify_artifact

BASE_MODEL_TYPE = "base"  # Registry key for artifacts


def load_base_model(version: str = None, map_location=None):
    """
    Load the full teacher model from artifacts.
    
    Args:
        version (str): Optional specific version to load.
        map_location: Optional torch device override.
    """
    if not verify_artifact(BASE_MODEL_TYPE):
        raise FileNotFoundError("Base model missing or corrupted in artifacts registry.")

    model_path = get_artifact_path(BASE_MODEL_TYPE)

    # If versioning is needed later, we'll resolve it here
    return load_torch_model(model_path, map_location or get_device())


# Optional: direct file path for local development (bypassing registry)
def get_local_base_path(filename="threat_model.pt"):
    """
    Returns the direct path to the base model in the package (dev only).
    """
    return Path(__file__).resolve().parent / filename 
