# -*- coding: utf-8 -*-
"""
Sentenial-X Health API
---------------------

Provides endpoints to monitor the health of the Sentenial-X system.

Features:
- API status
- Core simulator and telemetry health
- Optional extended checks for DB or engine status
"""

from __future__ import annotations

from fastapi import APIRouter
from datetime import datetime

from api.utils import api_response
from core.simulator import EmulationManager, TelemetryCollector

# ---------------------------------------------------------------------------
# Router setup
# ---------------------------------------------------------------------------
router = APIRouter(
    prefix="/health",
    tags=["health"]
)

# ---------------------------------------------------------------------------
# Core managers (singletons)
# ---------------------------------------------------------------------------
emulation_manager = EmulationManager()
telemetry_collector = TelemetryCollector()

# ---------------------------------------------------------------------------
# Health Endpoints
# ---------------------------------------------------------------------------

@router.get("/")
async def health_check():
    """
    Simple health check for the API
    """
    return api_response(
        {"status": "ok", "timestamp": datetime.utcnow().isoformat()},
        message="Sentenial-X API is online"
    )

@router.get("/simulators")
async def simulators_health():
    """
    Check health of registered simulators
    """
    status_list = []
    for sim in emulation_manager.simulators:
        try:
            # If run method exists, consider healthy
            healthy = callable(getattr(sim, "run", None))
            status_list.append({"simulator": sim.name, "healthy": healthy})
        except Exception:
            status_list.append({"simulator": sim.name, "healthy": False})

    return api_response(status_list, message=f"{len(status_list)} simulators checked")

@router.get("/telemetry")
async def telemetry_health():
    """
    Check health of the telemetry collector
    """
    try:
        last_entry = telemetry_collector.report(last_n=1)
        healthy = True
    except Exception:
        last_entry = None
        healthy = False

    return api_response(
        {"telemetry_last_entry": last_entry, "healthy": healthy},
        message="Telemetry collector status"
    )
