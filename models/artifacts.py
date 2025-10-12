# sentenialx/models/artifacts.py
import json
from pathlib import Path
from datetime import datetime
import hashlib
import logging

ARTIFACTS_DIR = Path(__file__).resolve().parent
REGISTRY_FILE = ARTIFACTS_DIR / "registry.json"
logging.basicConfig(level=logging.INFO)

def file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """Compute hash of a file."""
    hash_func = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def verify_model_integrity(file_path: Path, expected_hash: str) -> bool:
    """Verify file integrity against hash."""
    if not file_path.exists():
        return False
    computed_hash = file_hash(file_path)
    return computed_hash == expected_hash

def load_registry() -> dict:
    """Load the master registry."""
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_registry(data: dict):
    """Save the master registry."""
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logging.info("Registry updated.")

def get_artifact_path(model_type: str) -> Path:
    """Get path to artifact from registry."""
    reg = load_registry()
    if model_type not in reg:
        raise ValueError(f"No artifact entry for {model_type}")
    return ARTIFACTS_DIR / reg[model_type]["file"]

def verify_artifact(model_type: str) -> bool:
    """Verify artifact integrity."""
    reg = load_registry()
    if model_type not in reg:
        return False
    expected_hash = reg[model_type]["hash"]
    return verify_model_integrity(get_artifact_path(model_type), expected_hash)

def register_artifact(model_type: str, file_path: Path, version: str, metadata: dict | None = None):
    """Register a new artifact."""
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} not found.")
    hash_val = file_hash(file_path)
    rel_path = file_path.relative_to(ARTIFACTS_DIR)
    registry = load_registry()
    registry[model_type] = {
        "version": version,
        "file": str(rel_path),
        "hash": hash_val,
        "updated": datetime.utcnow().isoformat() + "Z",
        "metadata": metadata or {}
    }
    save_registry(registry)
    logging.info(f"Registered {model_type} v{version}")
    return registry
