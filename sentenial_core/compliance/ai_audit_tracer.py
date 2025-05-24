# sentenial_core/compliance/ai_audit_tracer.py

"""
AI Audit Tracer
---------------
Tracks AI decision-making processes to ensure auditability and regulatory compliance.
Records detailed trace logs mapping AI inferences to compliance standards.

Enables full transparency, accountability, and trace logs for forensic reviews.
"""

import json
import time
import uuid
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class AIAuditTracer:
    def __init__(self, audit_log_file: str = "audit_trace.log"):
        self.audit_log_file = audit_log_file

    def log_decision(self, decision_data: Dict[str, Any]):
        """
        Logs AI decision data with timestamp, trace_id, and compliance references.

        Args:
            decision_data (Dict[str, Any]): Arbitrary structured data capturing the decision context.
        """
        entry = {
            "trace_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "decision": decision_data
        }
        with open(self.audit_log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info(f"Logged AI decision trace with ID: {entry['trace_id']}")

    def load_audit_log(self) -> List[Dict[str, Any]]:
        """
        Loads and returns all audit log entries.

        Returns:
            List[Dict[str, Any]]: List of audit entries.
        """
        entries = []
        try:
            with open(self.audit_log_file, "r") as f:
                for line in f:
                    entries.append(json.loads(line.strip()))
        except FileNotFoundError:
            logger.warning("Audit log file not found.")
        return entries

    def find_trace_by_id(self, trace_id: str) -> Dict[str, Any]:
        """
        Finds a trace entry by its trace_id.

        Args:
            trace_id (str): Unique trace identifier.

        Returns:
            Dict[str, Any]: Trace entry or None.
        """
        logs = self.load_audit_log()
        for entry in logs:
            if entry.get("trace_id") == trace_id:
                return entry
        logger.info(f"Trace ID {trace_id} not found.")
        return {}

# Example usage
if __name__ == "__main__":
    tracer = AIAuditTracer()
    tracer.log_decision({
        "module": "regulatory_vector_matcher",
        "action": "matched regulation clause to control",
        "regulation": "GDPR Article 5",
        "control": "Data Encryption",
        "confidence": 0.92
    })
    all_logs = tracer.load_audit_log()
    print(f"Loaded {len(all_logs)} audit logs.")
