# Sentenial-X: The Ultimate Cyber Guardian

**Crafted for resilience. Engineered for vengeance.** Sentenial-X is not just a defense—it's a digital sentinel with the mind of a warrior and the reflexes of a machine.

Sentenial-X is an autonomous, modular cyber-defense platform designed for detection, triage, and safe, policy-driven response. It combines multi-modal machine learning (ML), real-time orchestration, and controlled emulation to enhance resilience and accelerate incident response.

- **License**: Apache-2.0 (with proprietary components; see [License & Contact](#license--contact) for details)
- **Status**: Alpha
- **Mode**: Beastmode

This README serves as a concise, developer-friendly entry point for contributors and operators. For enterprise deployments, refer to infrastructure code and hardened manifests in the `infra/` directory.

## Table of Contents

- [About](#about)
- [Key Features](#key-features)
- [Repository Layout](#repository-layout)
- [Architecture Overview](#architecture-overview)
- [Quickstart (Local Development)](#quickstart-local-development)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Security & Responsible Use](#security--responsible-use)
- [CI/CD & Infrastructure](#cicd--infrastructure)
- [Data & Models](#data--models)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License & Contact](#license--contact)
- [Appendix: Useful Commands](#appendix-useful-commands)

## About

Sentenial-X goes beyond passive defense by learning from telemetry and historical incidents, applying policy-controlled countermeasures, and providing immutable evidence for compliance and audits. It aims to detect, analyze, neutralize, and counter threats faster than attackers can pivot, while ensuring adaptability and self-healing capabilities.

The platform blends ML, orchestration, and governance for defensive and testing use cases, with a focus on enterprise security operations centers (SOCs). It supports cloud, on-premises, and hybrid environments.

## Key Features

- **Autonomous Detection & Response**: Policy-governed triage and actions, including quarantine, rerouting, and automatic recovery.
- **Adaptive AI**: Models improve with validated incident data using techniques like LoRA adapters and distillation.
- **Offensive-Defensive Fusion**: Controlled red-team emulation for resilience testing (e.g., pentest suite and ransomware simulator).
- **Compliance & Auditing**: Immutable, signed logs and reports compliant with NIST, ISO, GDPR, HIPAA, and DoD standards.
- **Resilience & Self-Healing**: Sandboxed execution (WASM/containers) for safe countermeasures.
- **Modular Components**:
  - **Cortex**: Natural language processing for threat analysis.
  - **Orchestrator**: AI-driven detection and response management.
  - **Simulator**: Attack scenario emulation for validation.
  - **Engine**: File protection, process monitoring, and network security.
  - **Dashboard**: Intuitive GUI for threats, compliance, and telemetry.

## Repository Layout

A high-level, developer-oriented structure (explore subfolders for detailed READMEs and CONTRIBUTING notes):

```
Sentenial-X/
├── apps/                      # Application services
│   ├── api-gateway/           # FastAPI routing, auth, observability
│   ├── dashboard/             # Next.js UI for SOC operators
│   ├── pentest-suite/         # Sandboxed red-team automation
│   └── ransomware-emulator/   # Safe simulation artifacts
├── sentenialx/                # Core ML and engine code
│   ├── src/                   # Source code
│   ├── models/                # Checkpoints and adapters (encoder, LoRA, distill, orchestrator)
│   ├── data/                  # Sanitized datasets (raw and processed)
│   └── docker/                # Container configurations
├── services/                  # Supporting services
│   ├── auth/                  # Authentication
│   ├── agent-manager/         # Telemetry collection and containment
│   ├── threat-engine/         # ML-based detection and scoring
│   ├── jailbreak-detector/    # Prompt-injection protection
│   ├── memory-core/           # Vector stores (FAISS/Chroma/Weaviate)
│   ├── compliance-engine/     # Reporting and evidence
│   ├── countermeasure-agent/  # Policy-driven actions
│   └── legal-shield/          # Legal templates and approvals
├── libs/                      # Shared utilities
│   ├── core/                  # Core libraries
│   ├── ml/                    # ML plugins
│   └── plugins/               # Extensions
├── infra/                     # Production deployments
│   ├── docker/                # Dockerfiles and Compose
│   ├── k8s/                   # Kubernetes manifests
│   └── terraform/             # Infrastructure as code
├── tests/                     # Testing suites
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
├── scripts/                   # Utility scripts
├── .env.example               # Environment template
├── requirements.txt           # Python dependencies
├── package.json               # Node.js dependencies
└── README.md                  # This file
```

## Architecture Overview

```
┌─────────────┐    ┌────────────────┐    ┌───────────────┐
│ Event/Logs  │ ──▶ │ TypeScript     │ ──▶ │ WebSocket /   │
│ Sources     │      │ Agent          │      │ API           │
└─────────────┘      └────────────────┘      └───────────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Python AI Module  │
              │ (analyze_threat)  │
              └───────────────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Dashboard Widgets │
              └───────────────────┘
```

1. **Agents** stream encrypted telemetry to the Agent Manager.
2. **API Gateway** routes data to the Threat Engine and Memory Core.
3. **Threat Engine** scores, classifies, and recommends playbooks using ML and rules.
4. **Orchestrator** triggers policy-checked Countermeasure Agent, updates Dashboard, and logs to Compliance Hub.
5. Actions produce signed, immutable logs in `data/logs/` for forensics.

Core Components:
- **API Gateway**: Centralized routing and auth (FastAPI/gRPC).
- **Dashboard**: React/Next.js GUI for triage and insights.
- **Agent Manager**: Endpoint telemetry and containment.
- **Threat Engine**: Multi-modal LLM for detection.
- **Jailbreak Detector**: Adversarial input protection.
- **Memory Core**: Encrypted vector retrieval.
- **Countermeasure Agent**: Sandboxed responses.
- **Pentest Suite**: Gated emulation tools.
- **Compliance & Legal Shield**: Automated evidence and reports.

## Quickstart (Local Development)

**Note**: For development/experimentation only. Do not run offensive modules on production or third-party systems without authorization.

### Prerequisites
- Docker & Docker Compose v2+
- Python 3.10+
- Node.js 18+

### Steps
1. Clone the repository:
   ```
   git clone git@github.com:erikg713/Sentenial-X.git
   cd Sentenial-X
   ```
2. Set up environment:
   ```
   cp .env.example .env
   # Edit: DATABASE_URL, REDIS_URL, VECTOR_DB_URL, OAUTH_CLIENTS, SECRET_KEYS
   ```
3. Start core infrastructure:
   ```
   docker compose up --build -d
   ```
4. Run API Gateway:
   ```
   cd apps/api-gateway
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
5. Start Dashboard:
   ```
   cd apps/dashboard
   npm install
   npm run dev
   # Access at http://localhost:3000
   ```
6. Run unit tests:
   ```
   pytest tests/unit -q
   ```

For production, use `infra/terraform` and `infra/helm`.

## Development Workflow

- **Python Services**: Use `libs/core` utilities. Enforce formatting with Black, linting with Ruff, and typing with Mypy.
- **Frontend**: Next.js + TypeScript; components in `apps/dashboard`.
- **ML Experiments**: Run in isolated containers; artifacts in model stores (not in repo).
- **Pull Requests**: One feature per PR, include tests, changelog, and architecture notes.

## Testing

- **Unit Tests**: `tests/unit/`
- **Integration Tests**: `tests/integration/` (use mocks and fixtures)
- **Model Evaluation**: `sentenialx/tests/test_llm_accuracy.py` (run in inference container)

## Security & Responsible Use

Sentenial-X includes emulation and response capabilities requiring strict governance:
- **Policy-First**: Countermeasures require policy checks and operator approval (test modes only).
- **RBAC**: Fine-grained roles for operators, auditors, developers.
- **Auditing**: Cryptographically signed logs for non-repudiation.
- **Legal Review**: Gated modules need approvals and NDAs.
- **Sandboxing**: WASM/containers for execution.
- **Safe Defaults**: Observe/emulate mode; destructive actions opt-in.

Obtain written authorization before running emulations on non-test systems. Consult legal/operations teams.

## CI/CD & Infrastructure

- **CI**: `.github/workflows/ci.yml` – linting, tests, builds, security scans (SAST/dependencies).
- **CD**: Terraform + Helm for cloud deployments (AWS/GCP/Azure).
- **Secrets**: Use Vault/SSM/KeyVault; avoid repo storage.
- **Observability**: Optional Prometheus/Grafana/ELK via `infra/helm`.

## Data & Models

- **data/raw/**: Inbound feeds (e.g., CVEs); sanitize before use.
- **data/processed/**: Tokenized JSONL for training/evaluation.
- **models/**: Checkpoints, adapters (LoRA/distillation); store in secure registries (S3/GCS).

Example: Run orchestrator:
```
python -m sentenialx.models.orchestrator.orchestrate --stage package
```

## Roadmap

1. **Alpha**: Core pipeline, API, agents, basic threat engine, dashboard.
2. **Beta**: LLM triage, jailbreak detector, sandboxed countermeasures.
3. **Enterprise**: Compliance automation, hardened infra, SOC integrations.
4. **Defense**: Certifications (STIGs), DoD workflows, offline runners.
5. **Futures**: On-device inference, hardware anchors, VR/AR UI.

## Contributing

We welcome vetted contributors. Before starting:
- Read `CONTRIBUTING.md` (conventions, commit rules).
- Sign NDAs for data/model access.
- Open issues with ATT&CK-mapped repro steps.
- Include tests and architecture notes in PRs.

## License & Contact

Sentenial-X is proprietary with Apache-2.0 elements. For licensing, trials, or access:
- Security inquiries: security@yourorg.example
- Development/sales: sales@yourorg.example

## Appendix: Useful Commands

- Lint & Format:
  ```
  ruff src/ services/ libs/ --fix
  black .
  ```
- Run Unit Tests:
  ```
  pytest tests/unit -q
  ```
- Build Docker Image (example):
  ```
  docker build -t sentenialx/api-gateway:alpha -f infra/docker/api-gateway.Dockerfile .
  ```
- Start Local Inference:
  ```
  docker compose -f docker/docker-compose.yml up --build inference
  ```
