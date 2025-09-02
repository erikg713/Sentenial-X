"""
Sentenial-X AI Core: Detection Module
-------------------------------------
Handles real-time threat detection, anomaly scoring, and attack pattern recognition
for the AI Core system. Integrates with DataStore for persistent logging and metrics.

Author: Sentenial-X Development Team
"""

import uuid
import time
from typing import Dict, Any, List
from datetime import datetime
from api.utils.logger import init_logger
from ai_core.datastore import get_datastore

logger = init_logger("ai_core.detection")


class DetectionEvent:
    """
    Represents a single detection event.
    """
    def __init__(self, source: str, event_type: str, data: Dict[str, Any], severity: str = "medium"):
        self.event_id = str(uuid.uuid4())
        self.source = source
        self.event_type = event_type
        self.data = data
        self.severity = severity
        self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source": self.source,
            "event_type": self.event_type,
            "data": self.data,
            "severity": self.severity,
            "timestamp": self.timestamp,
        }


class DetectionEngine:
    """
    Core detection engine for AI Core.
    Provides methods for threat scoring, anomaly detection, and event logging.
    """

    _HIGH_RISK_TYPES = {"sql_injection", "rce", "xss", "privilege_escalation"}

    def __init__(self):
        self._datastore = get_datastore()
        self._events: List[DetectionEvent] = []
        logger.info("DetectionEngine initialized")

    def detect(self, source: str, event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze incoming event and compute risk score.
        Stores event in memory and logs it to the datastore.
        """
        severity = self._compute_severity(event_type, data)
        event = DetectionEvent(source, event_type, data, severity)
        self._events.append(event)
        self._log_event(event)
        logger.info(f"Detection event: {event.to_dict()}")
        return event.to_dict()

    def _compute_severity(self, event_type: str, data: Dict[str, Any]) -> str:
        """
        Compute event severity based on type and content.
        Returns: "low", "medium", or "high"
        """
        if event_type in self._HIGH_RISK_TYPES:
            return "high"
        if any("admin" in str(v).lower() for v in data.values()):
            return "medium"
        return "low"

    def _log_event(self, event: DetectionEvent):
        """
        Persist detection event to datastore.
        """
        try:
            self._datastore.log_inference(
                log_id=event.event_id,
                model_id="detection_engine",
                input_data=event.data,
                output_data={"severity": event.severity, "event_type": event.event_type},
                confidence=1.0
            )
            logger.info(f"Detection event logged to datastore: {event.event_id}")
        except Exception as e:
            logger.error(f"Failed to log detection event: {e}")

    def get_events(self, severity_filter: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve stored detection events. Optional filtering by severity.
        """
        if severity_filter:
            return [e.to_dict() for e in self._events if e.severity.lower() == severity_filter.lower()]
        return [e.to_dict() for e in self._events]

    def clear_events(self):
        """
        Clear all in-memory detection events.
        """
        logger.warning(f"Clearing {len(self._events)} detection events")
        self._events.clear()


# ------------------------
# Quick CLI Test
# ------------------------
if __name__ == "__main__":
    engine = DetectionEngine()
    test_events = [
        {"source": "Cortex", "event_type": "sql_injection", "data": {"query": "DROP TABLE users;"}},
        {"source": "Orchestrator", "event_type": "scan", "data": {"tool": "nmap"}},
        {"source": "Cortex", "event_type": "xss", "data": {"payload": "<script>alert(1)</script>"}},
    ]

    for ev in test_events:
        result = engine.detect(ev["source"], ev["event_type"], ev["data"])
        print(result)

    print("All high severity events:", engine.get_events(severity_filter="high"))
