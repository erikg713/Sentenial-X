# Generate a full run.py implementation that performs semantic threat detection,
# logs zero-day SQLi attempts, and blocks IPs to a blocklist

from pathlib import Path

run_py_code = '''
import os
import json
import datetime
import random

# Simulated real-time request with semantic threat
def mock_request():
    return {
        "ip": "192.168.1.50",
        "payload": "'; DROP TABLE users; --",
        "vector": "sql_injection",
        "headers": {"User-Agent": "exploit-bot/1.0"},
        "zero_day": True
    }

# Detect and classify threat
def detect_threat(request):
    if "drop table" in request["payload"].lower():
        return {
            "type": request["vector"],
            "zero_day": request.get("zero_day", False),
            "severity": "critical",
            "ip": request["ip"],
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        }
    return None

# Block IP by appending to local blocklist
def block_ip(ip):
    with open("firewall/blocklist.txt", "a") as f:
        f.write(ip + "\\n")
    print(f"[BLOCK] {ip} added to blocklist.")

# Append to log file
def log_threat(threat):
    log_path = "logs/threats.json"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    try:
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                data = json.load(f)
        else:
            data = []
    except:
        data = []

    data.append(threat)
    with open(log_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Semantic Threat: {threat['type']}, Zero-Day: {threat['zero_day']}")

def main():
    os.makedirs("firewall", exist_ok=True)
    request = mock_request()
    threat = detect_threat(request)
    if threat:
        log_threat(threat)
        block_ip(threat["ip"])

if __name__ == "__main__":
    main()
'''

# Save run.py to workspace
run_path = "/mnt/data/run.py"
Path(run_path).write_text(run_py_code)

run_path
