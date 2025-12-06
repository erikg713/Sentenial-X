```markdown
# Cortex: Real-Time Threat Intelligence NLP Engine

**Cortex** is the high-performance Natural Language Processing (NLP) engine powering **Sentenial-X**, a next-generation autonomous cyber-defense platform. It continuously analyzes logs, telemetry, alerts, and unstructured text to detect attacker intent, classify threat behaviors, and enrich multi-modal fusion in the Threat Engine.

Cortex operates as a microservice with real-time streaming (Kafka/WebSocket), a REST API, interactive GUI, and full container orchestration — all integrated into Sentenial-X’s adaptive AI ecosystem.

- **License**: Apache-2.0 (with select proprietary components — see [License & Contact](#license--contact))
- **Status**: Production-Ready (Alpha → Beta transition Q4 2025)
- **Role**: Core component of the Sentenial-X Threat Engine

---

## Key Features

| Feature                        | Description |
|-------------------------------|-----------|
| **Threat Intent Classification** | Detects MITRE ATT&CK®-aligned tactics from raw logs using distilled, LoRA-tuned models |
| **Real-Time Streaming**         | Zero-latency processing via Kafka or WebSocket with backpressure handling |
| **FastAPI REST Interface**      | High-throughput `/predict` endpoint for synchronous analysis |
| **Interactive GUI**             | Real-time visualization, triage dashboard, and model explainability |
| **Centralized Artifact Registry** | All models, metadata, and hashes managed in `sentenialx/models/artifacts/` |
| **Integrity Verification**     | SHA-256 + registry validation before every model load |
| **Hot Model Reloading**         | Zero-downtime updates when new artifacts are registered |
| **Observability**               | Prometheus metrics, structured JSON logs, health checks |
| **Container-Native**            | Official Dockerfile + Helm chart ready |

---

## Architecture & Integration

```
Telemetry Sources
        ↓
API Gateway → Cortex (NLP Classification)
                        ↓
               Threat Engine (Multi-Modal Fusion)
                        ↓
                 Orchestrator → Countermeasure Agent
```

Cortex is **not** a standalone tool — it is the NLP brain of Sentenial-X:
- Consumes raw events from agents
- Outputs structured intent labels and confidence scores
- Feeds directly into scoring, playbook selection, and autonomous response

All models are versioned and cryptographically verified via the **central artifact registry** at `sentenialx/models/artifacts/`.

---

## Project Structure

```text
cortex/
├── cli.py                     → Training & runtime entrypoint
├── sentenial_x/
│   └── core/
│       └── cortex/
│           ├── daemon.py      → Background streaming service
│           ├── server.py      → FastAPI inference server
│           ├── gui.py         → Real-time visualization dashboard
│           ├── models/
│           │   └── inference.py → Model wrapper with registry integration
│           └── utils/         → Data loaders, preprocessing
├── datasets/
│   └── threat_intents.csv    → Sanitized training corpus
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

**Model artifacts are NOT stored here** — they live in the shared canonical registry:

```
sentenialx/models/artifacts/
├── registry.json              ← Master index (version, path, hash)
├── distill/threat_student_v1.onnx
├── lora/lora_weights_v1.bin
├── encoder/text_encoder_v1.pt
└── ... + .json metadata + .sha256 manifests
```

---

## Model Artifact Management (Central Registry)

All model assets are managed centrally for consistency, integrity, and zero-trust deployment.

### Registry Format (`registry.json`)
```json
{
  "distill": {
    "version": "1.0.0",
    "file": "distill/threat_student_v1.onnx",
    "hash": "e3b0c44298fc1c14...",
    "updated": "2025-08-15T14:22:00Z"
  }
}
```

### Helper API (`sentenialx/models/artifacts/__init__.py`)
- `get_artifact_path("distill")` → returns verified Path
- `verify_artifact("distill")` → SHA-256 check before load
- `register_artifact(...)` → called by orchestrator after training

**No hard-coded paths** — every component pulls from the registry.

---

## Quick Start

### 1. Train a New Model
```bash
python cortex/cli.py train datasets/threat_intents.csv
```
→ Automatically registers new version in central artifact registry.

### 2. Run Real-Time Processor
```bash
# Kafka mode (default)
python cortex/cli.py run --mode kafka --kafka localhost:9092 --topic sentenial_logs

# WebSocket mode
python cortex/cli.py run --mode websocket --ws ws://agent:8080/stream
```

### 3. Start API Server
```bash
uvicorn sentenial_x.core.cortex.server:app --host 0.0.0.0 --port 8080
```

### 4. Launch GUI
```bash
python -m sentenial_x.core.cortex.gui
```

### 5. Docker Compose (Full Stack)
```bash
docker compose up --build
```

---

## API Usage

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '["Failed password for root from 185.220.101.12", "User admin uploaded shell.php"]'
```

Response:
```json
[
  {"intent": "initial_access", "confidence": 0.98, "labels": {"...": 0.01}},
  {"intent": "persistence", "confidence": 0.95, "labels": {"...": 0.03}}
]
```

---

## Docker & Orchestration

```yaml
services:
  cortex:
    build: ./cortex
    ports:
      - "8080:8080"   # API
      - "8000:8000"   # Prometheus metrics
    volumes:
      - ./sentenialx/models/artifacts:/app/sentenialx/models/artifacts:ro
    environment:
      - CORTEX_MODE=kafka
      - KAFKA_BOOTSTRAP=kafka:9092
```

---

## Contributing

See `CONTRIBUTING.md` in the root repository.

All model contributions must:
- Be registered via `register_artifact()`
- Include training logs and evaluation report
- Pass integrity verification in CI

---

## License & Contact

- **Open Components**: Apache-2.0
- **Proprietary Components**: Restricted (model weights, certain adapters)

For access, collaboration, or enterprise licensing:

**Security & Research**: security@sentenialx.ai  
**Engineering**: cortex-dev@sentenialx.ai

---

**Cortex — Turning raw logs into actionable threat intelligence, in real time.**
``` 

This README is now **professional, clear, and enterprise-ready** — perfect for internal documentation, partner onboarding, or selective open-source release. Let me know if you'd like a dark-mode version or PDF export!
