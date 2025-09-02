# api/routes/monitor.py

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, Dict, Any
import psutil
import socket
import time

from api.utils import logger, risk_score, siem, soar, timeu
from agents.telemetry import TelemetryAgent
from agents.trace_agent import TraceAgent

router = APIRouter(prefix="/monitor", tags=["Monitoring"])

telemetry_agent = TelemetryAgent()
trace_agent = TraceAgent()

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check for API service.
    """
    logger.log_info("Health check requested")
    return {"status": "ok", "timestamp": timeu.utc_now_iso()}

@router.get("/system")
async def system_status() -> Dict[str, Any]:
    """
    Returns system resource usage: CPU, memory, disk, network.
    """
    cpu_percent = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    net = psutil.net_io_counters()

    system_data = {
        "hostname": socket.gethostname(),
        "cpu_percent": cpu_percent,
        "memory": {
            "total": mem.total,
            "available": mem.available,
            "percent": mem.percent,
        },
        "disk": {
            "total": disk.total,
            "used": disk.used,
            "percent": disk.percent,
        },
        "network": {
            "bytes_sent": net.bytes_sent,
            "bytes_recv": net.bytes_recv,
        },
        "timestamp": timeu.utc_now_iso(),
    }

    logger.log_info("System status retrieved", extra=system_data)
    telemetry_agent.capture("system_status", system_data)

    return system_data

@router.get("/trace")
async def trace_request(
    target: Optional[str] = Query(None, description="Target host or IP for trace"),
) -> Dict[str, Any]:
    """
    Run a trace operation against a target host/IP.
    """
    if not target:
        raise HTTPException(status_code=400, detail="Target parameter is required")

    logger.log_info(f"Trace requested for {target}")
    result = trace_agent.trace(target)

    telemetry_agent.capture("trace_request", {"target": target, "result": result})
    siem.forward_event("trace", result)

    return {
        "target": target,
        "result": result,
        "risk": risk_score.calculate(result),
        "timestamp": timeu.utc_now_iso(),
    }

@router.get("/telemetry")
async def get_telemetry() -> Dict[str, Any]:
    """
    Retrieve collected telemetry data.
    """
    data = telemetry_agent.dump()
    logger.log_info("Telemetry dump retrieved", extra={"count": len(data)})
    return {"telemetry": data, "timestamp": timeu.utc_now_iso()}

@router.post("/incident/escalate")
async def escalate_incident(incident: Dict[str, Any]):
    """
    Escalate an incident to SOAR for automated response.
    """
    if not incident:
        raise HTTPException(status_code=400, detail="No incident data provided")

    logger.log_warning("Incident escalation requested", extra=incident)
    action = soar.escalate(incident)

    telemetry_agent.capture("incident_escalation", {"incident": incident, "action": action})
    siem.forward_event("incident_escalation", {"incident": incident, "action": action})

    return {
        "status": "escalated",
        "incident": incident,
        "action": action,
        "timestamp": timeu.utc_now_iso(),
    }
