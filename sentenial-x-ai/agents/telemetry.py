# agents/telemetry.py
import psutil
import time
from typing import Dict, Any
from api.utils.logger import init_logger

logger = init_logger("telemetry_agent")

class TelemetryAgent:
    """Collects and provides system telemetry metrics for Sentenial-X."""

    def __init__(self, sample_interval: float = 1.0):
        """
        Initialize the telemetry agent.
        :param sample_interval: Time between telemetry samples in seconds.
        """
        self.sample_interval = sample_interval
        logger.info(f"TelemetryAgent initialized with interval {self.sample_interval}s")

    def collect(self) -> Dict[str, Any]:
        """Collect current telemetry metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            net_io = psutil.net_io_counters()

            telemetry_data = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_total": memory.total,
                "disk_percent": disk.percent,
                "disk_total": disk.total,
                "network_bytes_sent": net_io.bytes_sent,
                "network_bytes_recv": net_io.bytes_recv,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            logger.debug(f"Telemetry collected: {telemetry_data}")
            return telemetry_data
        except Exception as e:
            logger.error(f"Failed to collect telemetry: {e}")
            return {}

    def run(self):
        """Continuously collect telemetry metrics (for background agent)."""
        logger.info("TelemetryAgent started running in background mode.")
        try:
            while True:
                data = self.collect()
                # Optionally: send to API, DB, or message queue
                # e.g., orchestrator.update_telemetry(data)
                time.sleep(self.sample_interval)
        except KeyboardInterrupt:
            logger.info("TelemetryAgent stopped.")
        except Exception as e:
            logger.exception(f"TelemetryAgent crashed: {e}")


# Example usage
if __name__ == "__main__":
    agent = TelemetryAgent(sample_interval=5)
    while True:
        telemetry = agent.collect()
        print(telemetry)
        time.sleep(5)
