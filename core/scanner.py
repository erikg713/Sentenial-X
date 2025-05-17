# core/scanner.py
from queue import Queue
import time
# core/scanner.py
from queue import Queue
import time
import os

threat_queue = Queue()

def scan_from_sample(file_path="data/samples/Sampledata"):
    if not os.path.exists(file_path):
        print(f"[ERROR] Sample data file not found: {file_path}")
        return

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 3:
                continue  # Skip invalid lines
            ip, threat_type, severity = parts
            timestamp = time.strftime("%H:%M:%S")
            threat_queue.put((timestamp, ip, threat_type, severity))
            time.sleep(1)  # simulate delay
            
threat_queue = Queue()

def scan_network():
    # Example simulated scan
    threats = [
        ("192.168.1.10", "SQL Injection", "High"),
        ("192.168.1.42", "Port Scan", "Low"),
        ("192.168.1.88", "Brute Force Login", "Critical"),
    ]
    for ip, threat_type, severity in threats:
        timestamp = time.strftime("%H:%M:%S")
        threat_queue.put((timestamp, ip, threat_type, severity))
        time.sleep(1)
