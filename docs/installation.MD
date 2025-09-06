# Installation

This document covers a clean, repeatable way to set up a development environment for Sentenial‑X. It includes recommended commands, platform notes, and common troubleshooting tips so you can get the server and NLP components running quickly.

Table of contents
- Quick start
- Requirements
- Create and activate a virtual environment
- Install Python dependencies
- Install spaCy model(s)
- Run the application (development)
- Running tests
- Troubleshooting & tips
- Recommended pinned dependencies

---

## Quick start (recommended)

1. Ensure you have Python 3.11+ installed:
```bash
python --version
# or
python3 --version
```

2. Create and activate a virtual environment, upgrade pip, then install dependencies:
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

3. Download the spaCy model:
```bash
python -m spacy download en_core_web_sm
```

4. Start the development server (example):
```bash
# replace `main:app` with the module and ASGI app object used in this repo
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Requirements

- Python 3.11 or newer (3.11+)
- pip (use the pip inside the virtual environment)
- Recommended: virtualenv or venv for isolating dependencies
- External Python packages:
  - spaCy (en_core_web_sm)
  - scikit-learn
  - numpy
  - fastapi
  - uvicorn
  - reportlab

The repository includes a `requirements.txt`. Prefer installing from that file so versions remain consistent.

---

## Create and activate a virtual environment

Using the standard library venv:
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (cmd.exe)
.venv\Scripts\activate.bat
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Upgrade packaging tools before installing:
```bash
python -m pip install --upgrade pip setuptools wheel
```

---

## Install Python dependencies

With the venv activated:
```bash
pip install -r requirements.txt
```

If you need to install one package at a time, do so with pip and prefer pinned versions where appropriate:
```bash
pip install "fastapi>=0.95,<1.0" "uvicorn[standard]>=0.22,<1.0" "spacy>=3.5,<4.0"
```

Platform notes:
- On Debian/Ubuntu you may need build tools for some wheels:
```bash
sudo apt update
sudo apt install build-essential python3-dev
```
- On macOS, ensure Xcode command line tools are installed:
```bash
xcode-select --install
```

---

## Install spaCy model(s)

Download the small English model used by the project:
```bash
python -m spacy download en_core_web_sm
```

Verify the model loads:
```bash
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('spaCy model OK:', nlp.meta.get('name'))"
```

If you want larger models for better accuracy (at the cost of disk/RAM), consider `en_core_web_trf` (transformer-based); see spaCy docs for installation and hardware requirements.

---

## Run the application (development)

Start the FastAPI/ASGI server with uvicorn. Replace `main:app` with the actual module path and app object if different in this repo:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Common options:
- `--reload` — auto-reload during development
- `--workers N` — run multiple workers for production (use a process manager or container orchestration for production)

If your project places the FastAPI app under a package (for example `app.main:app`), point uvicorn to that import path.

Environment variables
- If the project uses an `.env` file, load it (e.g., via python-dotenv) or export required variables:
```bash
export APP_ENV=development
export SECRET_KEY="your-secret"
```

---

## Running tests

If the repository has tests (pytest is common), run:
```bash
pip install -r requirements-dev.txt  # if present
pytest -q
```

---

## Troubleshooting & tips

- Permission errors on install:
  - Use a virtual environment; do not run pip as root.
- Binary wheel build failures:
  - On Linux/WSL: install `build-essential`, `python3-dev`, and other system dependencies.
- spaCy model not found at runtime:
  - Ensure the venv used to install the model is the one used to run the app, and that `python -m spacy download en_core_web_sm` succeeded without errors.
- Inconsistent dependency versions:
  - Use pinned versions in `requirements.txt` and consider using a lock file (pip-tools, Poetry, or pipenv) for reproducibility.
- Large memory usage for transformer-based spaCy models:
  - Use `en_core_web_sm` for development and smaller resource footprints; choose `en_core_web_trf` only if you have the RAM/CPU or GPU required.

If you encounter specific errors, include:
- exact command used
- full error output
- Python version (`python --version`)
- OS and architecture

---

## Recommended pinned dependencies (example)

This is an example snippet showing conservative pinned ranges. The repo's `requirements.txt` should be the source of truth; consider pinning to exact versions for deployments.

```text
fastapi>=0.95,<1.0
uvicorn[standard]>=0.22,<1.0
spacy>=3.5,<4.0
scikit-learn>=1.2,<2.0
numpy>=1.24,<2.0
reportlab>=3.6,<4.0
```

---

If you want, I can:
- Update or create a `requirements.txt` with pinned versions,
- Add a small `Makefile` or npm-style scripts to streamline common tasks (setup, run, test),
- Or produce instructions customized to your project's import path for running the server (I can inspect the repo and adjust `uvicorn` command accordingly).
