#!/bin/bash
# scripts/seed_payloads.sh
# Sentenial-X: Seed example payloads, logs, and network anomalies

set -e

PAYLOAD_DIR="./data/raw/payloads"
LOGS_DIR="./data/raw/logs"
HTTP_DIR="./data/raw/http"

# Create directories if they don't exist
mkdir -p "$PAYLOAD_DIR"
mkdir -p "$LOGS_DIR"
mkdir -p "$HTTP_DIR"

echo "[Seed Payloads] Creating example payloads..."

# ---------------- Example Malware Payloads ----------------
declare -A payloads
payloads["malware1"]="MALWARE_EXEC /tmp/malicious.exe"
payloads["malware2"]="DROP TABLE users; -- SQL Injection"
payloads["malware3"]="<script>alert('XSS')</script>"

for name in "${!payloads[@]}"; do
    file_path="$PAYLOAD_DIR/$name.txt"
    if [ ! -f "$file_path" ]; then
        echo "${payloads[$name]}" > "$file_path"
        echo "[Seed Payloads] Created $file_path"
    fi
done

# ---------------- Example System Logs ----------------
logs=(
    "User login failed from IP 192.168.1.10"
    "File download detected: /tmp/malware.exe"
    "Unusual POST payload detected on /api/upload"
    "Multiple failed SSH attempts from 10.0.0.15"
)

for i in "${!logs[@]}"; do
    file_path="$LOGS_DIR/log_$i.txt"
    if [ ! -f "$file_path" ]; then
        echo "${logs[$i]}" > "$file_path"
        echo "[Seed Payloads] Created log file $file_path"
    fi
done

# ---------------- Example HTTP Traffic ----------------
http_requests=(
    "GET /login HTTP/1.1 Host: example.com User-Agent: Mozilla/5.0"
    "POST /api/upload HTTP/1.1 Host: example.com Content-Length: 1024"
    "GET /admin HTTP/1.1 Host: example.com Cookie: sessionid=abcd1234"
    "POST /login HTTP/1.1 Host: example.com User-Agent: test-agent"
)

for i in "${!http_requests[@]}"; do
    file_path="$HTTP_DIR/request_$i.txt"
    if [ ! -f "$file_path" ]; then
        echo "${http_requests[$i]}" > "$file_path"
        echo "[Seed Payloads] Created HTTP request $file_path"
    fi
done

echo "[Seed Payloads] All payloads, logs, and HTTP requests have been seeded."
