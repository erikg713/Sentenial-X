"""
cli/telemetry.py

Sentenial-X Telemetry Module - async live telemetry streaming.
Streams events from agents, sensors, or logs with optional filtering.
"""

import asyncio
import random
import time
import json
from cli.memory_adapter import get_adapter
from cli.logger import default_logger

# Mock telemetry sources (replace with real sensor/agent integration)
TELEMETRY_SOURCES = {
    "network_monitor": ["conn_established", "conn_failed", "high_latency", "intrusion_attempt"],
    "endpoint_sensor": ["process_start", "process_stop", "file_modified", "malware_detected"],
    "app_logs": ["error", "warning", "info"]
}


class Telemetry:
    def __init__(self):
        self.mem = get_adapter()
        self.logger = default_logger

    async def stream(self, source: str, filter_expr: str = ""):
        """
        Async generator for telemetry events.

        :param source: Name of the telemetry source
        :param filter_expr: Optional string to filter events
        :yield: dict event with timestamp and source
        """
        if source not in TELEMETRY_SOURCES:
            self.logger.error(f"Telemetry source '{source}' not recognized.")
            raise ValueError(f"Unknown telemetry source: {source}")

        events = TELEMETRY_SOURCES[source]

        self.logger.info(f"Starting telemetry stream from '{source}' with filter '{filter_expr}'")
        while True:
            await asyncio.sleep(random.uniform(0.5, 2.0))  # simulate event arrival

            event_type = random.choice(events)
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            event = {
                "source": source,
                "event_type": event_type,
                "severity": self._get_severity(event_type),
                "timestamp": timestamp
            }

            # Apply simple filter (contains string)
            if filter_expr and filter_expr.lower() not in event_type.lower():
                continue

            # Log to memory
            await self.mem.log_telemetry(event)

            # Yield for CLI or other consumers
            yield event

    def _get_severity(self, event_type: str) -> str:
        """Simple severity mapping"""
        high_sev = ["malware_detected", "intrusion_attempt", "conn_failed"]
        med_sev = ["error", "high_latency", "process_stop"]
        if event_type in high_sev:
            return "high"
        elif event_type in med_sev:
            return "medium"
        else:
            return "low"
