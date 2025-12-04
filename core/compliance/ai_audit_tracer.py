# Sentenial-X/core/compliance/ai_audit_tracer.py

import datetime
import json
import os
import threading
from typing import Any, Dict, Iterable, List, Optional

class AIAuditTracer:
    """
    Lightweight audit tracer for AI-related compliance events.
    - Writes JSONL (one JSON object per line) for easy parsing.
    - Optional redaction of sensitive fields in metadata.
    - Minimal, thread-safe file appends with a lock.
    """

    def __init__(
        self,
        log_file: str = "audit_log.jsonl",
        redacted_fields: Optional[Iterable[str]] = None,
        utc_isoformat: bool = True
    ) -> None:
        self.log_file = log_file
        self.redacted_fields = set(redacted_fields or [])
        self.utc_isoformat = utc_isoformat
        self._lock = threading.Lock()
        # Ensure parent directory exists
        parent = os.path.dirname(os.path.abspath(self.log_file))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        # Ensure file exists
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", encoding="utf-8") as f:
                pass

    def trace_event(
        self,
        user_id: str,
        action: str,
        metadata: Optional[Dict[str, Any]] = None,
        severity: str = "INFO",
        event_id: Optional[str] = None
    ) -> None:
        """
        Append an audit event.
        - user_id: initiator or system identifier
        - action: description (e.g., "model_decision", "override", "data_access")
        - metadata: free-form contextual details
        - severity: INFO | WARNING | ERROR
        - event_id: optional external correlation id
        """
        event = {
            "timestamp": self._now(),
            "severity": severity.upper(),
            "user_id": str(user_id),
            "action": str(action),
            "event_id": str(event_id) if event_id is not None else None,
            "metadata": self._redact(metadata or {}),
            "version": "1.0"  # schema version for forward-compat
        }
        line = json.dumps(event, ensure_ascii=False)
        with self._lock:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def load_events(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Read events from the log file (most basic loader).
        - limit: if provided, returns only the last N events.
        """
        with self._lock:
            with open(self.log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

        if limit is not None and limit > 0:
            lines = lines[-limit:]

        events: List[Dict[str, Any]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # skip corrupted lines quietly to stay lightweight
                continue
        return events

    def filter_events(
        self,
        severity: Optional[str] = None,
        action: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Basic in-memory filters over loaded events."""
        events = self.load_events(limit=limit)
        if severity:
            sev = severity.upper()
            events = [e for e in events if e.get("severity") == sev]
        if action:
            events = [e for e in events if e.get("action") == action]
        if user_id:
            events = [e for e in events if e.get("user_id") == user_id]
        return events

    def _redact(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive fields in a shallow dict."""
        if not self.redacted_fields:
            return metadata
        redacted = {}
        for k, v in metadata.items():
            redacted[k] = "***REDACTED***" if k in self.redacted_fields else v
        return redacted

    def _now(self) -> str:
        """Return timestamp in ISO 8601, UTC by default."""
        dt = datetime.datetime.utcnow() if self.utc_isoformat else datetime.datetime.now()
        # Always include 'Z' when using UTC to avoid ambiguity
        iso = dt.replace(microsecond=0).isoformat()
        return iso + ("Z" if self.utc_isoformat else "")


# Minimal example usage
if __name__ == "__main__":
    tracer = AIAuditTracer(
        log_file="audit_log.jsonl",
        redacted_fields={"ssn", "account_number"}
    )

    tracer.trace_event(
        user_id="user-123",
        action="model_decision",
        metadata={"model_version": "v2.1", "confidence": 0.92, "ssn": "123-45-6789"},
        severity="INFO",
        event_id="evt-0001"
    )

    tracer.trace_event(
        user_id="auditor-42",
        action="manual_override",
        metadata={"reason": "risk_flag", "note": "requires human review"},
        severity="WARNING"
    )

    print("Last events:", tracer.load_events(limit=5))
    print("Warnings:", tracer.filter_events(severity="WARNING"))
