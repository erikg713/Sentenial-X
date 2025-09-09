#!/usr/bin/env bash
# Start the telemetry Kafka consumer
set -euo pipefail

KAFKA_BOOTSTRAP="${KAFKA_BOOTSTRAP:-localhost:9092}"
KAFKA_TOPIC="${KAFKA_TOPIC:-telemetry.events}"
KAFKA_GROUP="${KAFKA_GROUP:-sentenialx-feature-consumer}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

export KAFKA_BOOTSTRAP
export KAFKA_TELEMETRY_TOPIC="${KAFKA_TOPIC}"
export KAFKA_GROUP
export PYTHONUNBUFFERED=1

echo "[+] Activating venv if present..."
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "[+] Starting telemetry consumer (Kafka: ${KAFKA_BOOTSTRAP}, topic: ${KAFKA_TOPIC})"
python telemetry/ingest_kafka.py
