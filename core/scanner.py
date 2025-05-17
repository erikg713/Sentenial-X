# core/scanner.py
from queue import Queue
import time

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
