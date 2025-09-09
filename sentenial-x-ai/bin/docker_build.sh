#!/usr/bin/env bash
# Build key Docker images used in the project
set -euo pipefail

MODEL_IMAGE="${MODEL_IMAGE:-your-registry/sentenialx-modelserver:latest}"
PREFETCH_IMAGE="${PREFETCH_IMAGE:-your-registry/sentenial-prefetch:latest}"

echo "[+] Building model server image -> ${MODEL_IMAGE}"
docker build -f docker/Dockerfile.modelserver -t "${MODEL_IMAGE}" .

echo "[+] Building prefetch image -> ${PREFETCH_IMAGE}"
docker build -f infra/docker/Dockerfile.prefetch -t "${PREFETCH_IMAGE}" .

echo "[+] Docker build complete."
