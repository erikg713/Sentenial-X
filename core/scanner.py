# core/scanner.py

from queue import Queue
import time
import os
import random

# Shared queue for real-time threat reporting
threat_queue = Queue()

def scan_from_sample(file_path="data/samples/Sampledata"):
    """
    Loads threats from a static sample file and pushes them to the real-time threat queue.
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] Sample data file not found: {file_path}")
        return

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 3:
                continue  # Skip malformed lines
            ip, threat_type, severity = parts
            timestamp = time.strftime("%H:%M:%S")
            threat_queue.put((timestamp, ip, threat_type, severity))
            time.sleep(random.uniform(0.8, 2.0))  # simulate scan delay

def simulate_live_threats():
    """
    Simulates live threat data (for dev/demo).
    """
    fake_ips = ["192.168.0.1", "10.0.0.12", "172.16.4.55"]
    threat_types = ["SQL Injection", "Brute Force", "Port Scan", "Ransomware"]
    severities = ["Low", "Medium", "High", "Critical"]

    while True:
        ip = random.choice(fake_ips)
        threat = random.choice(threat_types)
        severity = random.choice(severities)
        timestamp = time.strftime("%H:%M:%S")
        threat_queue.put((timestamp, ip, threat, severity))
        time.sleep(random.uniform(1.0, 3.0))
