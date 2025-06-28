from queue import Queue
import time
import os
import random
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SentenialX")

# Shared queue for real-time threat reporting
threat_queue = Queue()

def scan_from_sample(file_path: str = "data/samples/Sampledata") -> None:
    """
    Loads threats from a static sample file and pushes them to the real-time threat queue.

    Args:
        file_path (str): Path to the file containing sample threat data.
    """
    if not os.path.exists(file_path):
        logger.error(f"Sample data file not found: {file_path}")
        return

    try:
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != 3:
                    logger.warning(f"Malformed line skipped: {line.strip()}")
                    continue  # Skip malformed lines

                ip, threat_type, severity = parts
                if not validate_ip(ip) or not validate_severity(severity):
                    logger.warning(f"Invalid data format skipped: {line.strip()}")
                    continue

                timestamp = time.strftime("%H:%M:%S")
                threat_queue.put((timestamp, ip, threat_type, severity))
                time.sleep(random.uniform(0.8, 2.0))  # simulate scan delay
    except Exception as e:
        logger.exception(f"An error occurred while scanning from sample: {e}")

def simulate_live_threats() -> None:
    """
    Simulates live threat data (for development/demo purposes).
    """
    fake_ips = ["192.168.0.1", "10.0.0.12", "172.16.4.55"]
    threat_types = ["SQL Injection", "Brute Force", "Port Scan", "Ransomware"]
    severities = ["Low", "Medium", "High", "Critical"]

    try:
        while True:
            ip = random.choice(fake_ips)
            threat = random.choice(threat_types)
            severity = random.choice(severities)
            timestamp = time.strftime("%H:%M:%S")
            threat_queue.put((timestamp, ip, threat, severity))
            time.sleep(random.uniform(1.0, 3.0))
    except KeyboardInterrupt:
        logger.info("Simulation stopped by user.")
    except Exception as e:
        logger.exception(f"An error occurred in simulate_live_threats: {e}")

def validate_ip(ip: str) -> bool:
    """
    Validates the IP address format.

    Args:
        ip (str): The IP address to validate.

    Returns:
        bool: True if the IP address is valid, False otherwise.
    """
    parts = ip.split(".")
    return len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)

def validate_severity(severity: str) -> bool:
    """
    Validates the severity level.

    Args:
        severity (str): The severity level to validate.

    Returns:
        bool: True if the severity level is valid, False otherwise.
    """
    valid_severities = {"Low", "Medium", "High", "Critical"}
    return severity in valid_severities
