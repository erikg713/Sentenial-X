import random
import time
from datetime import datetime
from sentenial_core.engine.alert_dispatcher import dispatch_alert
from sentenial_core.engine.incident_logger import log_incident

class ThreatEmulator:
    """
    Simulates real-world cyber threats for testing and emulation purposes.
    Designed to generate realistic threat data in a safe and controlled manner.
    """

    def __init__(self, mode: str = "default"):
        self.mode = mode
        self.attack_signatures = [
            {"type": "port_scan", "target": "192.168.1.10", "ports": [22, 80, 443]},
            {"type": "brute_force", "target": "admin@localhost", "attempts": 50},
            {"type": "sql_injection", "url": "http://localhost/login", "payload": "' OR '1'='1"},
            {"type": "ddos", "target": "public-api.sentenial-x.ai", "rate": "1500 rps"},
            {"type": "ransomware", "host": "lab-machine-03", "encryption_time": "20s"},
        ]

    def simulate(self, count: int = 5):
        for _ in range(count):
            signature = random.choice(self.attack_signatures)
            timestamp = datetime.utcnow().isoformat()
            event = {
                "timestamp": timestamp,
                "emulated": True,
                "source": "ThreatEmulator",
                "threat_type": signature["type"],
                "details": signature,
            }

            print(f"[ThreatEmulator] Simulating: {signature['type']} @ {timestamp}")
            log_incident(event)
            dispatch_alert(event)
            time.sleep(random.uniform(0.5, 1.5))

    def run_live_feed(self, duration: int = 60):
        """
        Continuously emulates threats in real time for a given duration (seconds).
        """
        end_time = time.time() + duration
        while time.time() < end_time:
            self.simulate(count=1)
