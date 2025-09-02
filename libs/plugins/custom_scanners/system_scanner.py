from .base_scanner import BaseScanner
from typing import Dict, Any

class SystemScanner(BaseScanner):
    """
    Scanner for system-level irregularities (processes, files, registry).
    """

    def scan(self, target: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        options = options or {}
        return {
            "target": target,
            "issues": ["unauthorized_service"] if "service" in target else [],
            "options": options,
            "status": "completed",
        }

    def describe(self) -> str:
        return "Detects system-level anomalies and unauthorized changes"
