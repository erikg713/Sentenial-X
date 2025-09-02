# api/controllers/traces_controller.py
from fastapi import APIRouter, Query
from typing import List, Dict, Any, Optional
from agents.trace_agent import TraceAgent, AttackEvent
from api.utils.response import success_response

router = APIRouter()
# Use a singleton TraceAgent instance for API access
trace_agent = TraceAgent(history_size=1000, threat_threshold=80)

@router.get("/events", summary="Get Trace Events")
async def get_trace_events(severity: Optional[str] = Query(None, description="Filter by severity (info, medium, high)")) -> Dict[str, Any]:
    """
    Retrieve all trace events.
    Optional severity filter.
    """
    events = trace_agent.get_events(severity_filter=severity)
    return success_response(f"Retrieved {len(events)} trace events", events)

@router.post("/log", summary="Log a Trace Event")
async def log_trace_event(
    source: str,
    event_type: str,
    severity: Optional[str] = "info",
    payload: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Log a new trace event manually via API.
    """
    event_id = trace_agent.log_event(source=source, event_type=event_type, severity=severity, data=payload)
    return success_response(f"Trace event logged with id {event_id}", {"event_id": event_id})

@router.get("/blocked", summary="Get Blocked Sources")
async def get_blocked_sources() -> Dict[str, Any]:
    """
    Retrieve currently blocked sources from the TraceAgent.
    """
    blocked = list(trace_agent.blocked_sources)
    return success_response(f"{len(blocked)} sources blocked", blocked)

@router.get("/adaptation-log", summary="Get Adaptation Log")
async def get_adaptation_log() -> Dict[str, Any]:
    """
    Retrieve the TraceAgent adaptation log showing automated defense actions.
    """
    log = trace_agent.adaptation_log
    return success_response(f"{len(log)} adaptation entries", log)
