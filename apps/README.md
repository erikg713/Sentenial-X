Alright — in Sentenial-X, models/artifacts should be the canonical store for:

All model weight files (.pt, .onnx, .bin)

Model metadata (.json)

Hash/integrity manifests

Training logs, charts, and distilled model reports


This gives you a central, versioned artifact hub instead of scattering models all over your project.


---

Proposed Directory Layout

sentenialx/
├── models/
│   ├── artifacts/
│   │   ├── distill/
│   │   │   ├── threat_student_v1.onnx
│   │   │   ├── threat_student_v1_meta.json
│   │   │   └── threat_student_v1.sha256
│   │   ├── lora/
│   │   │   ├── lora_weights_v1.bin
│   │   │   ├── lora_weights_v1_meta.json
│   │   │   └── lora_weights_v1.sha256
│   │   ├── encoder/
│   │   │   ├── text_encoder_v1.pt
│   │   │   ├── text_encoder_v1_meta.json
│   │   │   └── text_encoder_v1.sha256
│   │   └── registry.json   ← master index of all models
│   ├── utils.py
│   ├── ...


---

models/artifacts/registry.json

This acts like your local PyPI for models — the orchestrator updates it after training/distillation.

{
  "distill": {
    "version": "1.0.0",
    "file": "distill/threat_student_v1.onnx",
    "hash": "abc123deadbeef...",
    "updated": "2025-08-15T14:22:00Z"
  },
  "lora": {
    "version": "1.0.0",
    "file": "lora/lora_weights_v1.bin",
    "hash": "fedcba987654321...",
    "updated": "2025-08-15T14:22:00Z"
  },
  "encoder": {
    "version": "1.0.0",
    "file": "encoder/text_encoder_v1.pt",
    "hash": "0987654321abcdef...",
    "updated": "2025-08-15T14:22:00Z"
  }
}


---

Artifact Management Helper (artifacts.py)

# sentenialx/models/artifacts/__init__.py

import json
from pathlib import Path
from sentenialx.models.utils import file_hash, verify_model_integrity

ARTIFACTS_DIR = Path(__file__).resolve().parent
REGISTRY_FILE = ARTIFACTS_DIR / "registry.json"


def load_registry() -> dict:
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_registry(data: dict):
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_artifact_path(model_type: str) -> Path:
    reg = load_registry()
    if model_type not in reg:
        raise ValueError(f"No artifact entry for {model_type}")
    return ARTIFACTS_DIR / reg[model_type]["file"]


def verify_artifact(model_type: str) -> bool:
    reg = load_registry()
    if model_type not in reg:
        return False
    expected_hash = reg[model_type]["hash"]
    return verify_model_integrity(get_artifact_path(model_type), expected_hash)


def register_artifact(model_type: str, file_path: Path, version: str):
    hash_val = file_hash(file_path)
    rel_path = file_path.relative_to(ARTIFACTS_DIR)
    registry = load_registry()
    registry[model_type] = {
        "version": version,
        "file": str(rel_path),
        "hash": hash_val,
        "updated": __import__("datetime").datetime.utcnow().isoformat()
    }
    save_registry(registry)
    return registry


---

How It Fits the Bigger Picture

Orchestrator → runs LoRA fine-tuning or distillation → saves new weights into models/artifacts/{type}/...

Registry → updated with version, hash, file path

Pipeline → instead of hardcoding paths, calls get_artifact_path("distill")

Integrity checks → verify_artifact("distill") before loading



---

If you want, I can now update your build_inference_pipeline so it automatically pulls from models/artifacts registry, meaning you never have to manually change file paths when you release a new model.

Do you want me to wire it so the pipeline always uses the latest registered version?

