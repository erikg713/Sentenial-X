# -*- coding: utf-8 -*-
"""
Sentenial-X Threat API
----------------------

Provides REST endpoints for:
- Running threat simulators
- Collecting telemetry
- Managing attack playbooks
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from core.simulator import EmulationManager, TelemetryCollector
from core.simulator.wormgpt_clone import WormGPTDetector
from core.simulator.blind_spot_tracker import BlindSpotTracker
from core.simulator.attack_playbook import create_playbook

# ---------------------------------------------------------------------------
# Router setup
# ---------------------------------------------------------------------------
router = APIRouter(
    prefix="/threat",
    tags=["threat"]
)

# ---------------------------------------------------------------------------
# Core managers (singleton pattern)
# ---------------------------------------------------------------------------
emulation_manager = EmulationManager()
telemetry_collector = TelemetryCollector()

# Register default simulators
emulation_manager.register(WormGPTDetector())
emulation_manager.register(BlindSpotTracker())

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@router.get("/")
async def root():
    return {"message": "Sentenial-X Threat API is online"}

@router.get("/simulators")
async def list_simulators():
    """
    List all registered simulators
    """
    return {"simulators": [sim.name for sim in emulation_manager.simulators]}

@router.post("/simulate/{sim_name}")
async def run_simulator(sim_name: str):
    """
    Run a single simulator by name
    """
    simulator = next((s for s in emulation_manager.simulators if s.name == sim_name), None)
    if not simulator:
        raise HTTPException(status_code=404, detail=f"Simulator '{sim_name}' not found")
    
    result = simulator.run()
    telemetry_collector.add({"simulator": sim_name, "result": result})
    return {"simulator": sim_name, "result": result}

@router.get("/telemetry")
async def get_telemetry(last_n: Optional[int] = Query(10, description="Number of latest telemetry entries")):
    """
    Retrieve recent telemetry events
    """
    data = telemetry_collector.report(last_n=last_n)
    return {"telemetry": data}

@router.get("/telemetry/summary")
async def telemetry_summary():
    """
    Return summarized telemetry
    """
    return {"summary": telemetry_collector.summary()}

@router.get("/playbook")
async def generate_playbook():
    """
    Generate a sample attack playbook
    """
    playbook = create_playbook()
    return {
        "playbook_id": playbook.id,
        "name": playbook.name,
        "steps": [{"id": s.id, "description": s.description} for s in playbook.steps]
    }

@router.post("/playbook/run")
async def run_playbook():
    """
    Execute a sample playbook
    """
    playbook = create_playbook()
    results = []
    for step in playbook.steps:
        result = step.execute()
        telemetry_collector.add({"playbook_step": step.id, "result": result})
        results.append({"step_id": step.id, "result": result})
    return {"playbook_id": playbook.id, "results": results}
