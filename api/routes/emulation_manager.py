# -*- coding: utf-8 -*-
"""
Emulation Manager API Routes for Sentenial-X
--------------------------------------------

Exposes endpoints to register, start, stop, and run multiple simulators
through the EmulationManager, and collect telemetry.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.simulator import EmulationManager, TelemetryCollector, discover_simulators

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/emulation_manager", tags=["EmulationManager"])

manager = EmulationManager()
collector = TelemetryCollector()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ManagerAction(BaseModel):
    simulator_name: str


class ManagerResponse(BaseModel):
    status: str
    simulators: List[str]
    timestamp: datetime


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/register", response_model=ManagerResponse)
async def register_simulator(action: ManagerAction) -> ManagerResponse:
    """
    Register a simulator with the EmulationManager.
    """
    simulators = {s.name: s for s in discover_simulators()}
    if action.simulator_name not in simulators:
        raise HTTPException(status_code=404, detail=f"Simulator '{action.simulator_name}' not found")

    sim = simulators[action.simulator_name]
    manager.register(sim)
    logger.info("Registered simulator: %s", sim.name)

    return ManagerResponse(status="registered", simulators=[s.name for s in manager.simulators], timestamp=datetime.utcnow())


@router.post("/start_all", response_model=ManagerResponse)
async def start_all() -> ManagerResponse:
    """Start all registered simulators."""
    try:
        manager.start_all()
        return ManagerResponse(status="started", simulators=[s.name for s in manager.simulators], timestamp=datetime.utcnow())
    except Exception as e:
        logger.exception("Failed to start all simulators")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop_all", response_model=ManagerResponse)
async def stop_all() -> ManagerResponse:
    """Stop all registered simulators."""
    try:
        manager.stop_all()
        return ManagerResponse(status="stopped", simulators=[s.name for s in manager.simulators], timestamp=datetime.utcnow())
    except Exception as e:
        logger.exception("Failed to stop all simulators")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run_all", response_model=ManagerResponse)
async def run_all(sequential: bool = True) -> ManagerResponse:
    """
    Run all registered simulators. Can run sequentially or in parallel.
    """
    try:
        manager.run_all(sequential=sequential)
        return ManagerResponse(status="completed", simulators=[s.name for s in manager.simulators], timestamp=datetime.utcnow())
    except Exception as e:
        logger.exception("Failed to run all simulators")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/telemetry_summary", response_model=dict)
async def telemetry_summary() -> dict:
    """
    Return a summary of all collected telemetry from simulators.
    """
    return collector.summary()
