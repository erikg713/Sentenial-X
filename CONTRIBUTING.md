# CONTRIBUTING.md — Cortex & Sentenial-X
------------------------------------------
Thank you for your interest in contributing to **Cortex**, the real-time NLP threat intelligence engine powering **Sentenial-X**.
---

### This guide covers everything you need to know to contribute effectively — from code style to model submission, security requirements, and release process. ###

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Ways to Contribute](#ways-to-contribute)
- [Development Setup](#development-setup)
- [Code Style & Standards](#code-style--standards)
- [Branching Strategy](#branching-strategy)
- [Pull Request Process](#pull-request-process)
- [Model Contributions](#model-contributions-critical)
- [Security & Compliance](#security--compliance)
- [Testing Requirements](#testing-requirements)
- [Release Process](#release-process)
- [Contact & Support](#contact--support)

---

## Code of Conduct

All contributors must follow the **Sentenial-X Code of Conduct** (see `CODE_OF_CONDUCT.md` in root).  
Be respectful, inclusive, and professional.

---

## Ways to Contribute

| Type                    | Examples |
|-----------------------|--------|
| **Bug fixes**         | Crash in WebSocket reconnect, memory leak in batching |
| **Features**          | New MITRE ATT&CK intent classes, ONNX export support |
| **Performance**       | Faster embedding inference, lower CPU usage |
| **Documentation**     | Improve README, add examples, API docs |
| **Testing**           | Add integration tests, model accuracy benchmarks |
| **Model improvements**| Better distillation, LoRA adapters, new datasets |

We especially welcome **model contributions** that improve intent detection accuracy or reduce latency.

---

## Development Setup

```bash
# Clone the full monorepo (required for artifact registry access)
git clone https://github.com/erikg713/Sentenial-X.git
cd Sentenial-X
---

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
---
### Install Cortex + dev dependencies ###
```
pip install -r services/cortex/requirements.txt
pip install -e .  # if using editable install
pip install pre-commit black ruff mypy pytest
```
---
# Install pre-commit hooks #
```
pre-commit install
```

> **Never** commit model weights (`.pt`, `.onnx`, `.bin`) directly. Use the artifact registry only.

---

## Code Style & Standards

We enforce strict quality:

| Tool       | Command                     | Enforced On |
|-----------|-----------------------------|-----------|
| Black     | `black .`                   | CI        |
| Ruff      | `ruff check .`              | CI        |
| MyPy      | `mypy sentenial_x/`         | CI        |
| Prettier  | GUI/frontend files          | CI        |

Run locally:
```bash
pre-commit run --all-files
```

All new code must:
- Be fully type-annotated
- Include docstrings (Google or NumPy style)
- Have unit tests (>85% coverage for new modules)

---

## Branching Strategy

```text
main          ← Production (never commit directly)
release/x.y   ← Release branches
feature/xxx   ← Your work
hotfix/xxx    ← Critical fixes
```

Branch naming:
- `feature/nlp-new-intents`
- `bugfix/websocket-reconnect`
- `model/distill-v2`

---

## Pull Request Process

1. **Fork & clone** the repository
2. Create your feature branch
3. Write tests + documentation
4. Run full test suite:
   ```bash
   pytest tests/
   pre-commit run --all-files
   ```
5. Open PR against `main`
6. Fill the PR template (required)
7. Wait for review from **2 maintainers**
8. Address feedback → merge via **squash & merge**

**All PRs are scanned** for:
- Secrets
- Model files
- Backdoors
- Dependency vulnerabilities

---

## Model Contributions (CRITICAL)

**Never commit model weights directly.**

All models must go through the **central artifact registry**:

### How to Submit a New Model

1. Train your model (anywhere)
2. Save weights to a temporary location
3. Run the registration script:
   ```bash
   python scripts/register_model.py \
     --type distill \
     --file /tmp/threat_student_v2.onnx \
     --version 2.0.0 \
     --notes "Improved exfiltration detection +15% F1"
   ```
4. This will:
   - Compute SHA-256
   - Move file to `sentenialx/models/artifacts/distill/`
   - Update `registry.json`
   - Create metadata + training report

5. Open a PR with **only** the updated `registry.json` and report

**Approval requires**:
- Accuracy report (on holdout set)
- Latency benchmark (<50ms on CPU)
- No increase in false positives
- Signed-off training data provenance

---

## Security & Compliance

This project handles sensitive threat data.

You **must**:
- Never log real customer data
- Never commit credentials, IPs, or PII
- Use `log.debug()` only for non-sensitive info
- All external API calls must go through `core/adapters/api_adapter.py`

We run:
- GitGuardian secret scanning
- Bandit static analysis
- Trivy container scanning
- Model integrity verification on load

---

## Testing Requirements

| Type              | Command                     | Required |
|-------------------|-----------------------------|----------|
| Unit tests        | `pytest tests/unit/`        | Yes      |
| Integration       | `pytest tests/integration/` | Yes      |
| Model accuracy    | `python scripts/eval_model.py` | For model PRs |
| Performance       | `python scripts/benchmark.py`  | Recommended |

---

## Release Process

1. Version bump in `sentenial_x/__init__.py`
2. Update `registry.json` if models changed
3. Draft release with:
   - Model performance summary
   - Changelog
   - Docker image tags
4. Publish via GitHub Releases
5. Update Helm chart in `infra/charts/cortex`

---

## Contact & Support

| Purpose                  | Contact |
|--------------------------|-------|
| General questions        | cortex-dev@sentenialx.ai |
| Security disclosures     | security@sentenialx.ai (PGP available) |
| Model submission review | ai-research@sentenialx.ai |
| Slack (internal)         | `#cortex-dev` |

---

**Welcome to the team.**  
Your contribution helps protect critical infrastructure worldwide.

— The Sentenial-X Cortex Team
``` 
