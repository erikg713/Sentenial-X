# apps/dashboard/widgets/threat_panel.py

"""
Threat Panel Widget
------------------
Displays real-time threat events, alerts, and classified risks
from the Sentenial-X AI threat detection modules.
"""

from fastapi import APIRouter
from typing import List, Dict
from agents.trace_agent import TraceAgent, AttackEvent

router = APIRouter()
trace_agent = TraceAgent()

@router.get("/events")
async def recent_threats(severity: str = None) -> Dict[str, List[Dict]]:
    """
    Retrieve recent threat events.
    Optionally filter by severity (info, medium, high).
    """
    events = trace_agent.get_events(severity_filter=severity)
    return {
        "status": "success",
        "message": f"Retrieved {len(events)} threat events",
        "data": events
    }

@router.get("/blocked")
async def blocked_sources() -> Dict[str, List[str]]:
    """
    Retrieve a list of currently blocked sources.
    """
    blocked = list(trace_agent.blocked_sources)
    return {
        "status": "success",
        "message": f"{len(blocked)} sources are currently blocked",
        "data": blocked
    }

@router.get("/adaptation_log")
async def adaptation_log() -> Dict[str, List[Dict]]:
    """
    Retrieve the TraceAgent adaptation log for threat mitigation actions.
    """
    log = trace_agent.adaptation_log
    return {
        "status": "success",
        "message": f"{len(log)} adaptation actions logged",
        "data": log
    }

# Example usage:
# GET /widgets/threat_panel/events
# GET /widgets/threat_panel/blocked
# GET /widgets/threat_panel/adaptation_log
