# sentenialx/models/distill/__init__.py

from pathlib import Path
from sentenialx.models.utils import load_torch_model, get_device
from sentenialx.models.artifacts import get_artifact_path, verify_artifact

DISTILL_MODEL_TYPE = "distilled"  # Registry key for artifacts


def load_distilled_model(version: str = None, map_location=None):
    """
    Load the distilled (student) threat model from artifacts.

    Args:
        version (str): Optional specific version to load.
        map_location: Optional torch device override.
    """
    if not verify_artifact(DISTILL_MODEL_TYPE):
        raise FileNotFoundError("Distilled model missing or corrupted in artifacts registry.")

    model_path = get_artifact_path(DISTILL_MODEL_TYPE)
    return load_torch_model(model_path, map_location or get_device())


# Optional: dev-only shortcut
def get_local_distill_path(filename="threat_student.pt"):
    """
    Returns the direct path to the distilled model in the package (dev only).
    """
    return Path(__file__).resolve().parent / filename
