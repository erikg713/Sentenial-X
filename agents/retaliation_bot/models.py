from datetime import datetime
from typing import Dict, Any, Optional

class ThreatEvent:
    def __init__(self, source_ip: str, vector: str, severity: int, details: Optional[Dict[str, Any]] = None):
        self.source_ip = source_ip
        self.vector = vector
        self.severity = severity
        self.timestamp = datetime.utcnow()
        self.details = details or {}

    def __str__(self):
        return (f"[{self.timestamp}] Threat from {self.source_ip} - "
                f"Vector: {self.vector}, Severity: {self.severity}, Details: {self.details}")