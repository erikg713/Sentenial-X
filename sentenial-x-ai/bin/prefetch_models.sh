#!/usr/bin/env bash
# Prefetch models into a local directory using the prefetch container or script.
set -euo pipefail

# Host folder to mount where models will be stored
MODELS_HOST_DIR="${MODELS_HOST_DIR:-/mnt/models}"
PREFETCH_IMAGE="${PREFETCH_IMAGE:-your-registry/sentenial-prefetch:latest}"
DOCKER_RUNTIME="${DOCKER_RUNTIME:-docker}"  # or podman

echo "[+] Ensuring host models directory exists: ${MODELS_HOST_DIR}"
mkdir -p "${MODELS_HOST_DIR}"

if command -v "${DOCKER_RUNTIME}" >/dev/null 2>&1; then
  echo "[+] Running prefetch container (${PREFETCH_IMAGE}) mounting ${MODELS_HOST_DIR}:/opt/models"
  "${DOCKER_RUNTIME}" run --rm --gpus all -v "${MODELS_HOST_DIR}:/opt/models" "${PREFETCH_IMAGE}"
else
  echo "[!] Docker not found. Trying local script fallback..."
  if [ -f "infra/docker/prefetch_models.py" ]; then
    python infra/docker/prefetch_models.py
  else
    echo "[ERROR] No prefetch method available. Install Docker or add a local prefetch script."
    exit 2
  fi
fi

echo "[+] Prefetch complete. Models available at ${MODELS_HOST_DIR}"
