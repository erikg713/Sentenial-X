# apps/dashboard/widgets/countermeasure_log.py

"""
Countermeasure Log Widget
------------------------
Provides a live view of all countermeasures executed by Sentenial-X agents.
Logs include timestamp, action, target source, and reason for the countermeasure.
"""

from fastapi import APIRouter
from typing import List, Dict
from agents.trace_agent import TraceAgent

router = APIRouter()
trace_agent = TraceAgent()

@router.get("/logs")
async def countermeasure_logs(severity: str = None) -> Dict[str, List[Dict]]:
    """
    Retrieve all countermeasure logs.
    Optional filter by severity (e.g., "high", "medium", "low").
    """
    logs = trace_agent.adaptation_log
    if severity:
        logs = [log for log in logs if log.get("severity", "").lower() == severity.lower()]
    return {"status": "success", "message": "Countermeasure logs fetched", "data": logs}

# Example usage:
# GET /widgets/countermeasure_log/logs
# GET /widgets/countermeasure_log/logs?severity=high
