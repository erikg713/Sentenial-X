# -*- coding: utf-8 -*-
"""
Orchestrator API Routes for Sentenial-X
---------------------------------------

Exposes endpoints to execute orchestrator commands for automated
policy enforcement, configuration updates, and system-level tasks.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.simulator.orchestrator import Orchestrator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/orchestrator", tags=["Orchestrator"])

orchestrator = Orchestrator()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class OrchestratorRequest(BaseModel):
    action: str
    params: Dict[str, Any] = {}


class OrchestratorResponse(BaseModel):
    action: str
    status: str
    result: Any
    executed_at: datetime


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/execute", response_model=OrchestratorResponse)
async def execute_command(request: OrchestratorRequest) -> OrchestratorResponse:
    """
    Execute an orchestrator action with optional parameters.
    """
    try:
        result = orchestrator.execute(request.action, **request.params)
        logger.info("Executed orchestrator action: %s", request.action)
        return OrchestratorResponse(
            action=request.action,
            status="success",
            result=result,
            executed_at=datetime.utcnow()
        )
    except Exception as e:
        logger.exception("Orchestrator execution failed")
        raise HTTPException(status_code=500, detail=f"Orchestrator error: {e}")
