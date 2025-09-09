# agents/endpoint_agent.py

import os
import time
import socket
import logging
import psutil
import platform
import threading
import requests
from datetime import datetime
from typing import Dict, Any, Optional

# Local imports from Sentenial-X
from core.engine.file_integrity import FileIntegrityMonitor
from core.engine.process_inspector import ProcessInspector
from core.engine.network_watcher import NetworkWatcher
from core.engine.alert_dispatcher import AlertDispatcher
from core.engine.incident_logger import IncidentLogger

logger = logging.getLogger("SentenialX.EndpointAgent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class EndpointAgent:
    """
    Production-ready endpoint agent for Sentenial-X.
    Runs as a lightweight monitoring daemon, collecting telemetry,
    scanning for anomalies, reporting to orchestrator, and applying countermeasures.
    """

    def __init__(self,
                 orchestrator_url: str,
                 agent_id: Optional[str] = None,
                 poll_interval: int = 10,
                 heartbeat_interval: int = 60):
        self.orchestrator_url = orchestrator_url.rstrip("/")
        self.agent_id = agent_id or socket.gethostname()
        self.poll_interval = poll_interval
        self.heartbeat_interval = heartbeat_interval

        # Core components
        self.file_monitor = FileIntegrityMonitor()
        self.process_monitor = ProcessInspector()
        self.network_monitor = NetworkWatcher()
        self.alert_dispatcher = AlertDispatcher()
        self.incident_logger = IncidentLogger()

        # Control flags
        self.running = False

    def collect_system_telemetry(self) -> Dict[str, Any]:
        """
        Collect basic system stats for health reporting.
        """
        try:
            cpu = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            boot_time = datetime.fromtimestamp(psutil.boot_time()).isoformat()

            telemetry = {
                "agent_id": self.agent_id,
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "cpu_percent": cpu,
                "memory_used": mem.used,
                "memory_total": mem.total,
                "disk_used": disk.used,
                "disk_total": disk.total,
                "boot_time": boot_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            return telemetry
        except Exception as e:
            logger.error(f"Error collecting system telemetry: {e}")
            return {}

    def send_heartbeat(self):
        """
        Periodically send a heartbeat to the orchestrator.
        """
        while self.running:
            try:
                telemetry = self.collect_system_telemetry()
                url = f"{self.orchestrator_url}/agents/heartbeat"
                response = requests.post(url, json=telemetry, timeout=10)
                if response.status_code == 200:
                    logger.info(f"Heartbeat sent successfully for {self.agent_id}")
                else:
                    logger.warning(f"Heartbeat failed with status {response.status_code}")
            except Exception as e:
                logger.error(f"Failed to send heartbeat: {e}")
            time.sleep(self.heartbeat_interval)

    def monitor_loop(self):
        """
        Main loop for continuous endpoint monitoring.
        """
        while self.running:
            try:
                # File integrity check
                file_alerts = self.file_monitor.scan()
                for alert in file_alerts:
                    self._handle_alert("file_integrity", alert)

                # Process inspection
                proc_alerts = self.process_monitor.scan()
                for alert in proc_alerts:
                    self._handle_alert("process", alert)

                # Network monitoring
                net_alerts = self.network_monitor.scan()
                for alert in net_alerts:
                    self._handle_alert("network", alert)

                time.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(self.poll_interval)

    def _handle_alert(self, source: str, alert: Dict[str, Any]):
        """
        Dispatch alerts to orchestrator and log incidents.
        """
        try:
            logger.warning(f"[{source.upper()} ALERT] {alert}")
            self.incident_logger.log(alert)

            payload = {
                "agent_id": self.agent_id,
                "source": source,
                "alert": alert,
                "timestamp": datetime.utcnow().isoformat()
            }

            url = f"{self.orchestrator_url}/alerts"
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info(f"Alert dispatched to orchestrator: {alert}")
            else:
                logger.warning(f"Failed to dispatch alert, status={response.status_code}")

            # Optionally trigger countermeasure locally
            if alert.get("severity", "low") in ["high", "critical"]:
                self._apply_countermeasure(alert)

        except Exception as e:
            logger.error(f"Failed to handle alert: {e}")

    def _apply_countermeasure(self, alert: Dict[str, Any]):
        """
        Example local countermeasure logic.
        """
        try:
            if alert.get("type") == "malicious_process":
                pid = alert.get("pid")
                if pid:
                    psutil.Process(pid).kill()
                    logger.info(f"Killed malicious process {pid}")
            elif alert.get("type") == "unauthorized_connection":
                conn = alert.get("connection")
                logger.info(f"Would block unauthorized connection: {conn}")
            # Extend with more countermeasures
        except Exception as e:
            logger.error(f"Countermeasure failed: {e}")

    def start(self):
        """
        Start the endpoint agent service.
        """
        logger.info(f"Starting Endpoint Agent: {self.agent_id}")
        self.running = True

        threading.Thread(target=self.monitor_loop, daemon=True).start()
        threading.Thread(target=self.send_heartbeat, daemon=True).start()

    def stop(self):
        """
        Stop the agent.
        """
        logger.info("Stopping Endpoint Agent")
        self.running = False


if __name__ == "__main__":
    # Example standalone run
    agent = EndpointAgent(orchestrator_url="http://localhost:8000")
    try:
        agent.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        agent.stop()
