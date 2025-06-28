"""
controller.py

This module manages the execution of scanner-related tasks using threading.
"""

import threading
import logging
from core import scanner

# Configure logging
logging.basicConfig(level=logging.INFO)

# List to manage threads
threads = []

def start_thread(target_function):
    """
    Starts a thread for the given target function and manages it in the threads list.
    """
    try:
        thread = threading.Thread(target=target_function, daemon=True)
        threads.append(thread)
        thread.start()
        logging.info(f"Started thread for {target_function.__name__}.")
    except Exception as e:
        logging.error(f"Error starting thread for {target_function.__name__}: {e}")

def run_sample_threats():
    """
    Launches a thread to perform scanning from sample threats.
    """
    start_thread(scanner.scan_from_sample)

def run_simulated_threats():
    """
    Launches a thread to simulate live threats.
    """
    start_thread(scanner.simulate_live_threats)

def run_actual_scan():
    """
    Placeholder for running actual scans using tools like Nmap or other vulnerability scanners.
    """
    raise NotImplementedError("run_actual_scan() is not implemented yet.")

def join_threads():
    """
    Ensures all threads are joined before exiting.
    """
    for thread in threads:
        if thread.is_alive():
            thread.join()

# Future: Add more scanning methods and integrate advanced tools.
