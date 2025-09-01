"""
telemetry.py
-------------
Telemetry and monitoring module for the chain-executor subsystem
of Sentenial-X. Handles structured logging, execution metrics,
performance tracking, and distributed telemetry export.

Author: Sentenial-X Dev Team
"""

import os
import time
import json
import socket
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import psutil  # For system stats
import uuid


class Telemetry:
    """
    Telemetry class for monitoring chain execution performance,
    errors, and distributed telemetry reporting.
    """

    def __init__(self, service_name: str = "chain-executor", export_path: Optional[str] = None):
        self.service_name = service_name
        self.hostname = socket.gethostname()
        self.export_path = export_path or os.getenv("CHAIN_EXECUTOR_TELEMETRY", "logs/chain_executor_telemetry.json")

        # Set up logging
        os.makedirs(os.path.dirname(self.export_path), exist_ok=True)
        logging.basicConfig(
            filename=self.export_path.replace(".json", ".log"),
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )

        logging.info(f"[Telemetry] Initialized for {self.service_name} on {self.hostname}")

    def _system_metrics(self) -> Dict[str, Any]:
        """Collect system-level telemetry stats."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.5),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage("/").percent,
            "active_processes": len(psutil.pids()),
        }

    def record_event(self, event_type: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Records a telemetry event.
        """
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "service": self.service_name,
            "hostname": self.hostname,
            "event_type": event_type,
            "details": details,
            "system_metrics": self._system_metrics(),
        }

        # Save to JSON file
        self._export_event(entry)
        logging.info(f"Telemetry event recorded: {event_type} - {details}")

        return entry

    def record_execution(self, task_id: str, status: str, duration: float, metadata: Optional[Dict[str, Any]] = None):
        """
        Record a chain execution event.
        """
        return self.record_event(
            "execution",
            {
                "task_id": task_id,
                "status": status,
                "duration_sec": duration,
                "metadata": metadata or {},
            },
        )

    def record_error(self, task_id: str, error_message: str, stacktrace: Optional[str] = None):
        """
        Record an error during chain execution.
        """
        return self.record_event(
            "error",
            {
                "task_id": task_id,
                "error_message": error_message,
                "stacktrace": stacktrace,
            },
        )

    def record_performance(self, task_id: str, latency_ms: float, throughput: Optional[float] = None):
        """
        Record performance metrics for a given execution task.
        """
        return self.record_event(
            "performance",
            {
                "task_id": task_id,
                "latency_ms": latency_ms,
                "throughput": throughput,
            },
        )

    def _export_event(self, entry: Dict[str, Any]):
        """
        Append telemetry event to JSON file for auditing and analytics.
        """
        try:
            if not os.path.exists(self.export_path):
                with open(self.export_path, "w") as f:
                    json.dump([entry], f, indent=4)
            else:
                with open(self.export_path, "r+") as f:
                    data = json.load(f)
                    data.append(entry)
                    f.seek(0)
                    json.dump(data, f, indent=4)
        except Exception as e:
            logging.error(f"[Telemetry] Failed to export telemetry event: {e}")


# Example usage
if __name__ == "__main__":
    telemetry = Telemetry()

    # Record a fake execution
    start_time = time.time()
    time.sleep(1.2)  # simulate execution delay
    duration = time.time() - start_time

    telemetry.record_execution("task-123", "success", duration, {"step": "graph_node_eval"})
    telemetry.record_performance("task-123", latency_ms=duration * 1000, throughput=10.5)
    telemetry.record_error("task-456", "NullPointerException in node execution", "Traceback...")

    print("Telemetry demo events recorded.")
