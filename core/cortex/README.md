# Cortex: Real-Time Threat Intelligence NLP Engine

Cortex is a high-performance Natural Language Processing (NLP) engine integrated into the Sentenial-X cyber-defense platform. It specializes in detecting and classifying threat intents from system logs, telemetry, and other textual data sources. Supporting real-time streaming via Kafka and WebSocket, a REST API for integration, a graphical user interface (GUI) for visualization, and containerized deployments, Cortex enhances the Threat Engine's multi-modal analysis capabilities. It leverages adaptive AI models managed through Sentenial-X's centralized artifact registry for versioned, secure model handling.

- **License**: Apache-2.0 (with proprietary components; see [License & Contact](#license--contact) for details)
- **Status**: Alpha
- **Integration**: Part of Sentenial-X's modular framework, contributing to threat triage and adaptive learning.

This README provides a developer-friendly guide for setup, usage, and integration. For broader Sentenial-X context, refer to the main repository README.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Model Artifact Management](#model-artifact-management)
- [Training](#training)
- [Real-Time Streaming](#real-time-streaming)
- [Background Service](#background-service)
- [API Server](#api-server)
- [GUI](#gui)
- [Docker Deployment](#docker-deployment)
- [Example API Request](#example-api-request)
- [Contributing](#contributing)
- [License & Contact](#license--contact)

## Features

- **Threat Intent Classification**: Analyzes log data for malicious patterns using NLP models, integrating with Sentenial-X's Threat Engine for scoring and playbook recommendations.
- **Real-Time Streaming**: Supports Kafka and WebSocket for continuous data ingestion and processing.
- **REST API**: FastAPI-based endpoint for seamless integration with external systems or other Sentenial-X components.
- **GUI**: Interactive dashboard for visualization, model insights, and manual triage.
- **Dockerized Deployment**: Scalable, containerized setup with optional GPU support for inference.
- **Adaptive AI Integration**: Models evolve via Sentenial-X's artifact registry, using LoRA fine-tuning and distillation for efficiency.
- **Security & Compliance**: Sandboxed execution, immutable logging, and policy-governed operations aligned with NIST/ISO/GDPR/HIPAA/DoD.

## Installation

Clone the repository and install dependencies:

```
git clone https://github.com/erikg713/Sentenial-X.git
cd Sentenial-X/apps/cortex  # Assuming Cortex is placed here; adjust as needed
pip install -r requirements.txt
```

For production, use Docker as described below.

## Project Structure

A streamlined structure focused on modularity:

```
cortex/
├── cli.py                  # Command-line interface for training and running
├── __init__.py
├── datasets/
│   └── threat_intents.csv  # Example training dataset (sanitized logs)
├── sentenial_x/
│   └── core/
│       └── cortex/
│           ├── daemon.py   # Background service for streaming
│           ├── server.py   # FastAPI + Uvicorn API server
│           ├── gui.py      # GUI launcher
│           ├── models/     # Model-related code; artifacts managed centrally
│           ├── utils/      # Helper utilities (e.g., data loaders)
│           └── __init__.py
├── docker-compose.yml      # Multi-service composition
├── Dockerfile              # Container build
├── README.md               # This file
└── requirements.txt        # Dependencies (e.g., FastAPI, Uvicorn, Kafka libs)
```

Models are not stored here; they are pulled from `sentenialx/models/artifacts/` for centralized management.

## Configuration

Configure via environment variables, a `.env` file, or CLI arguments. Key parameters:

| Parameter     | Description                          | Default                  |
|---------------|--------------------------------------|--------------------------|
| `--mode`      | Run mode: `kafka` or `websocket`     | `kafka`                  |
| `--topic`     | Kafka topic to consume               | `pinet_logs`             |
| `--kafka`     | Kafka broker address                 | `localhost:9092`         |
| `--ws`        | WebSocket endpoint                   | `ws://localhost:8080/logs` |
| `--host`      | API server host                      | `0.0.0.0`                |
| `--port`      | API server port                      | `8080`                   |
| `DATA_PATH`   | Path to training dataset             | `datasets/threat_intents.csv` |
| `MODEL_TYPE`  | Model type to load (e.g., `distill`) | `distill`                |

For Docker, load `.env` during build/compose.

## Model Artifact Management

Cortex uses Sentenial-X's centralized artifact system for models, ensuring versioned, hashed, and verifiable storage. Artifacts are stored in `sentenialx/models/artifacts/`, with a `registry.json` as the master index.

### Key Artifacts
- Model weights (`.pt`, `.onnx`, `.bin`)
- Metadata (`.json`)
- Integrity hashes (`.sha256`)
- Training logs, charts, and reports

Example `registry.json`:
```json
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
```

### Artifact Helper (`artifacts.py`)
Integrated into Cortex's inference pipeline:
```python
# sentenialx/models/artifacts/__init__.py (excerpt)

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
```

### Updated Inference Pipeline
Cortex's build_inference_pipeline now pulls from the registry:
- Load model: `path = get_artifact_path(MODEL_TYPE)`
- Verify: `if not verify_artifact(MODEL_TYPE): raise IntegrityError`
- Post-training: Save weights to `artifacts/{type}/`, then `register_artifact(type, path, version)`
- Orchestrator integration: Runs fine-tuning/distillation, updates registry automatically.

This centralizes models, avoiding scattered files.

## Training

Train models using CLI:
```
python cortex/cli.py train --data datasets/threat_intents.csv
```
Outputs are registered in the artifact system. Use sanitized data from `data/processed/`.

## Real-Time Streaming

- **Kafka**:
  ```
  python cortex/cli.py run --mode kafka --topic pinet_logs --kafka localhost:9092
  ```
- **WebSocket**:
  ```
  python cortex/cli.py run --mode websocket --ws ws://localhost:8080/logs
  ```

Streams feed into NLP analysis, with results routed to the Threat Engine.

## Background Service

Run as a daemon:
```
python -m sentenial_x.core.cortex.daemon --mode kafka --topic pinet_logs --kafka localhost:9092
```

## API Server

Launch the REST API:
```
uvicorn sentenial_x.core.cortex.server:app --host 0.0.0.0 --port 8080
```

## Example API Request

Predict threat intent:
```
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Suspicious login attempt detected"}'
```

## GUI

Launch the interactive GUI:
```
python -m sentenial_x.core.cortex.gui
```
Visualizes threats, model performance, and real-time telemetry.

## Docker Deployment

- **Build**:
  ```
  docker build -t sentenialx-cortex .
  ```
- **Run (with GPU)**:
  ```
  docker run --gpus all --rm -it sentenialx-cortex
  ```
- **Compose**:
  ```
  docker compose up --build
  ```

Mount volumes for artifacts and datasets.

## Contributing

Follow Sentenial-X's `CONTRIBUTING.md`. Include tests, architecture notes, and sign NDAs for model access. Open issues with ATT&CK mappings.

## License & Contact

Proprietary with Apache-2.0 elements. Contact:
- Security: security@yourorg.example
- Sales/Dev: sales@yourorg.example
