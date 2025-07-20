import time
import psutil
import threading
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger("BlindSpotTracker")
logger.setLevel(logging.INFO)

class BlindSpotTracker:
    def __init__(self):
        self.running = False
        self.anomalies: list[Dict[str, Any]] = []

    def _scan(self):
        logger.info("Blind Spot Tracker scan started.")
        while self.running:
            try:
                self.detect_unusual_processes()
                self.detect_unlinked_binaries()
                self.detect_high_cpu_invisible_processes()
                time.sleep(10)  # periodic scan
            except Exception as e:
                logger.error(f"[BlindSpotTracker] Error during scan: {e}")

    def detect_unusual_processes(self):
        for proc in psutil.process_iter(attrs=["pid", "name", "exe", "username"]):
            try:
                if proc.info['exe'] is None or proc.info['exe'] == "":
                    anomaly = {
                        "type": "Unlinked Executable",
                        "pid": proc.info["pid"],
                        "process": proc.info["name"],
                        "user": proc.info["username"],
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    self.anomalies.append(anomaly)
                    logger.warning(f"⚠️ Blind spot: {anomaly}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    def detect_unlinked_binaries(self):
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if not proc.info['cmdline']:
                    anomaly = {
                        "type": "Missing Command Line",
                        "pid": proc.info['pid'],
                        "process": proc.info['name'],
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    self.anomalies.append(anomaly)
                    logger.warning(f"⚠️ Blind spot: {anomaly}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    def detect_high_cpu_invisible_processes(self, cpu_threshold: float = 20.0):
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
            try:
                cpu_usage = proc.cpu_percent(interval=0.1)
                if cpu_usage > cpu_threshold and not proc.name():
                    anomaly = {
                        "type": "High CPU Ghost Process",
                        "pid": proc.pid,
                        "cpu": cpu_usage,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    self.anomalies.append(anomaly)
                    logger.warning(f"⚠️ Blind spot: {anomaly}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._scan, daemon=True)
            self.thread.start()
            logger.info("Blind Spot Tracker started.")

    def stop(self):
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join()
            logger.info("Blind Spot Tracker stopped.")

    def get_anomalies(self) -> list:
        return self.anomalies