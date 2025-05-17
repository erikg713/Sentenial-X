# core/controller.py

import threading
from core import scanner

def run_sample_threats():
    thread = threading.Thread(target=scanner.scan_from_sample, daemon=True)
    thread.start()

def run_simulated_threats():
    thread = threading.Thread(target=scanner.simulate_live_threats, daemon=True)
    thread.start()

# Future: run_actual_scan(), integrate Nmap or vulnerability scanner
