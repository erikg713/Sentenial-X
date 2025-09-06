# -*- coding: utf-8 -*-
"""
Attack Playbook API Routes for Sentenial-X
------------------------------------------

Exposes endpoints to create, list, and run attack playbooks,
allowing orchestration of multiple simulation steps.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.simulator.attack_playbook import Playbook, create_playbook, execute_playbook, list_playbooks

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/playbooks", tags=["Playbooks"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PlaybookRequest(BaseModel):
    name: Optional[str] = None
    steps: Optional[List[str]] = None


class PlaybookResponse(BaseModel):
    id: str
    name: str
    steps: List[str]
    status: str
    executed_at: Optional[datetime] = None


class PlaybookListResponse(BaseModel):
    total: int
    playbooks: List[PlaybookResponse]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/create", response_model=PlaybookResponse)
async def create_new_playbook(request: PlaybookRequest) -> PlaybookResponse:
    """
    Create a new attack playbook.
    """
    try:
        pb = create_playbook(name=request.name, steps=request.steps)
        logger.info("Created playbook: %s", pb.name)
        return PlaybookResponse(
            id=pb.id,
            name=pb.name,
            steps=[s.id for s in pb.steps],
            status="created"
        )
    except Exception as e:
        logger.exception("Failed to create playbook")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=PlaybookListResponse)
async def list_all_playbooks() -> PlaybookListResponse:
    """
    List all existing attack playbooks.
    """
    try:
        pbs: List[Playbook] = list_playbooks()
        response = [
            PlaybookResponse(
                id=pb.id,
                name=pb.name,
                steps=[s.id for s in pb.steps],
                status="ready"
            )
            for pb in pbs
        ]
        return PlaybookListResponse(total=len(response), playbooks=response)
    except Exception as e:
        logger.exception("Failed to list playbooks")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute/{playbook_id}", response_model=PlaybookResponse)
async def execute_existing_playbook(playbook_id: str) -> PlaybookResponse:
    """
    Execute a playbook by its ID.
    """
    try:
        pb = execute_playbook(playbook_id)
        logger.info("Executed playbook: %s", pb.name)
        return PlaybookResponse(
            id=pb.id,
            name=pb.name,
            steps=[s.id for s in pb.steps],
            status="executed",
            executed_at=datetime.utcnow()
        )
    except Exception as e:
        logger.exception("Failed to execute playbook")
        raise HTTPException(status_code=500, detail=str(e))
