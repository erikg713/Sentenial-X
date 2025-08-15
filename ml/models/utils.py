# sentenialx/models/utils.py

import hashlib
import json
import torch
from pathlib import Path
from datetime import datetime


def get_device(prefer_gpu: bool = True):
    """Select the best available device."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    elif prefer_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_model_path(*parts) -> Path:
    """
    Resolve a model file path inside sentenialx/models/.
    Example:
        resolve_model_path("distill", "threat_student.pt")
    """
    base_dir = Path(__file__).resolve().parent
    return base_dir.joinpath(*parts)


def save_model_metadata(model_name: str, version: str, extra: dict = None):
    """
    Save metadata for a model in JSON format.
    """
    meta = {
        "model": model_name,
        "version": version,
        "timestamp": datetime.utcnow().isoformat(),
        "device": str(get_device()),
    }
    if extra:
        meta.update(extra)

    meta_path = resolve_model_path(f"{model_name}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta_path


def load_model_metadata(model_name: str) -> dict:
    """
    Load model metadata if it exists.
    """
    meta_path = resolve_model_path(f"{model_name}_meta.json")
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def file_hash(path: Path, algo: str = "sha256") -> str:
    """
    Compute a hash for a given file.
    """
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_model_integrity(path: Path, expected_hash: str) -> bool:
    """
    Verify that the model file matches the expected hash.
    """
    return file_hash(path) == expected_hash


def load_torch_model(path: Path, map_location=None):
    """
    Load a PyTorch model with device mapping.
    """
    device = map_location or get_device()
    return torch.load(path, map_location=device)


def save_torch_model(model, path: Path):
    """
    Save a PyTorch model.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path
