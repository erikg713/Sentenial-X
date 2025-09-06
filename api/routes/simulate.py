# -*- coding: utf-8 -*-
"""
Simulation API Routes for Sentenial-X
-------------------------------------

Exposes endpoints to:
- List available simulators
- Run a simulation (single or playbook)
- Collect telemetry from simulations
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from core.simulator import (
    discover_simulators,
    EmulationManager,
    TelemetryCollector,
)
from core.simulator.attack_playbook import create_playbook

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/simulate",
    tags=["Simulation"],
    responses={404: {"description": "Not found"}},
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class SimulatorInfo(BaseModel):
    name: str
    description: str


class SimulationResult(BaseModel):
    simulator: str
    status: str
    timestamp: datetime
    telemetry: Optional[dict] = None


class PlaybookResult(BaseModel):
    id: str
    name: str
    executed: List[str]
    telemetry_summary: dict


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/list", response_model=List[SimulatorInfo])
async def list_simulators() -> List[SimulatorInfo]:
    """
    List all available simulators.
    """
    simulators = discover_simulators()
    return [SimulatorInfo(name=s.name, description=s.description) for s in simulators]


@router.post("/run", response_model=SimulationResult)
async def run_simulation(
    simulator_name: str = Query(..., description="Name of the simulator to run")
) -> SimulationResult:
    """
    Run a single simulator by name.
    """
    manager = EmulationManager()
    simulators = {s.name: s for s in discover_simulators()}

    if simulator_name not in simulators:
        raise HTTPException(status_code=404, detail=f"Simulator '{simulator_name}' not found")

    sim = simulators[simulator_name]
    manager.register(sim)

    telemetry_collector = TelemetryCollector()
    manager.run_all(sequential=True)

    return SimulationResult(
        simulator=sim.name,
        status="completed",
        timestamp=datetime.utcnow(),
        telemetry=telemetry_collector.summary(),
    )


@router.post("/playbook", response_model=PlaybookResult)
async def run_playbook() -> PlaybookResult:
    """
    Run the default attack playbook.
    """
    playbook = create_playbook()
    manager = EmulationManager()
    telemetry_collector = TelemetryCollector()

    for sim in playbook.steps:
        manager.register(sim)

    manager.run_all(sequential=True)

    return PlaybookResult(
        id=playbook.id,
        name=playbook.name,
        executed=[s.id for s in playbook.steps],
        telemetry_summary=telemetry_collector.summary(),
    )
