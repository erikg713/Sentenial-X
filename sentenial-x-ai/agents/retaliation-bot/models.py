from datetime import datetime
from typing import Dict, Any, Optional
from typing import List, Optional
import re

class SimpleThreatModel:
    """
    Lightweight rule-based or ML-ready model for threat classification.
    Returns threat categories that map to countermeasures.
    """

    THREAT_KEYWORDS = {
        "malware": [r"malware", r"trojan", r"ransomware", r"virus", r".*\.exe"],
        "sql_injection": [r"drop table", r"union select", r"insert into", r"sql injection"],
        "xss": [r"<script>", r"alert\(", r"onerror=", r"xss"],
    }

    def __init__(self):
        pass

    def classify(self, logs: List[str]) -> List[Optional[str]]:
        """
        Classify a batch of logs into threat categories.
        Returns None if no threat detected (normal activity).
        """
        results = []
        for log in logs:
            threat = None
            log_lower = log.lower()
            for category, patterns in self.THREAT_KEYWORDS.items():
                for pattern in patterns:
                    if re.search(pattern, log_lower):
                        threat = category
                        break
                if threat:
                    break
            results.append(threat)
        return results
        
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
