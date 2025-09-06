#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# Seed initial attack payloads into the Sentenial-X database
# ---------------------------------------------------------
# Usage:
#   ./seed_payloads.sh
#   ./seed_payloads.sh --dry-run

set -euo pipefail

# -------------------------------
# Configuration (env overrides)
# -------------------------------
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-sentenialx}"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:-password}"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  echo "⚠️  Running in dry-run mode. No changes will be applied."
fi

# -------------------------------
# Payloads to seed
# -------------------------------
declare -a PAYLOADS=(
  '{"id": "p001", "name": "Phishing Email", "type": "email", "description": "Simulated phishing email with malicious link"}'
  '{"id": "p002", "name": "Ransomware Dropper", "type": "malware", "description": "File-based ransomware payload"}'
  '{"id": "p003", "name": "SQL Injection Test", "type": "web", "description": "Test SQL injection attack payload"}'
  '{"id": "p004", "name": "Brute Force SSH", "type": "network", "description": "Simulated SSH brute force attempt"}'
)

# -------------------------------
# Seed function
# -------------------------------
function seed_payload() {
  local payload_json="$1"
  local id name type description

  id=$(echo "$payload_json" | jq -r '.id')
  name=$(echo "$payload_json" | jq -r '.name')
  type=$(echo "$payload_json" | jq -r '.type')
  description=$(echo "$payload_json" | jq -r '.description')

  if [ "$DRY_RUN" = true ]; then
    echo "[Dry-Run] Would insert payload: $id | $name | $type | $description"
  else
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c \
      "INSERT INTO attack_payloads (id, name, type, description) VALUES ('$id', '$name', '$type', '$description') ON CONFLICT (id) DO NOTHING;"
    echo "✅ Inserted payload: $id | $name"
  fi
}

# -------------------------------
# Main seeding loop
# -------------------------------
echo "🌐 Seeding attack payloads into $DB_NAME..."
for payload in "${PAYLOADS[@]}"; do
  seed_payload "$payload"
done

echo "✅ Seeding complete."
