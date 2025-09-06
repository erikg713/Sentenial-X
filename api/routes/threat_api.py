# -*- coding: utf-8 -*-
"""
Threat API Routes for Sentenial-X
---------------------------------

These routes provide endpoints for managing threat simulations,
fuzzing, worm emulation, and blind spot tracking.
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional

from core.simulator import wormgpt_clone, synthetic_attack_fuzzer, blind_spot_tracker

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/threats",
    tags=["Threats"],
    responses={404: {"description": "Not found"}},
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ThreatRequest(BaseModel):
    """Request model for launching a threat simulation."""
    name: str
    parameters: Optional[dict] = None


class ThreatResponse(BaseModel):
    """Response model for threat execution results."""
    success: bool
    message: str
    details: Optional[dict] = None


class ThreatListResponse(BaseModel):
    """Response model for listing available threats."""
    threats: List[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/", response_model=ThreatListResponse)
async def list_threats() -> ThreatListResponse:
    """
    List all available threat simulators.
    """
    threats = ["wormgpt_clone", "synthetic_attack_fuzzer", "blind_spot_tracker"]
    return ThreatListResponse(threats=threats)


@router.post("/run", response_model=ThreatResponse)
async def run_threat(request: ThreatRequest) -> ThreatResponse:
    """
    Run a specified threat simulator with given parameters.
    """
    try:
        if request.name == "wormgpt_clone":
            result = wormgpt_clone.run(request.parameters or {})
        elif request.name == "synthetic_attack_fuzzer":
            result = synthetic_attack_fuzzer.run(request.parameters or {})
        elif request.name == "blind_spot_tracker":
            result = blind_spot_tracker.run(request.parameters or {})
        else:
            raise HTTPException(status_code=400, detail="Unknown threat name")

        return ThreatResponse(success=True, message="Threat executed", details=result)
    except Exception as e:
        logger.exception("Threat execution failed")
        raise HTTPException(status_code=500, detail=f"Execution error: {e}")


@router.post("/stop", response_model=ThreatResponse)
async def stop_threat(
    name: str = Query(..., description="Name of the running threat simulation to stop")
) -> ThreatResponse:
    """
    Stop a running threat simulator.
    """
    try:
        if name == "wormgpt_clone":
            wormgpt_clone.stop()
        elif name == "synthetic_attack_fuzzer":
            synthetic_attack_fuzzer.stop()
        elif name == "blind_spot_tracker":
            blind_spot_tracker.stop()
        else:
            raise HTTPException(status_code=400, detail="Unknown threat name")

        return ThreatResponse(success=True, message=f"Threat '{name}' stopped")
    except Exception as e:
        logger.exception("Failed to stop threat")
        raise HTTPException(status_code=500, detail=f"Stop error: {e}")
