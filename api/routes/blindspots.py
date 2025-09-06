# -*- coding: utf-8 -*-
"""
Blind Spot Detection API Routes for Sentenial-X
-----------------------------------------------

Exposes endpoints to detect gaps in coverage or detection blind spots
in monitoring, simulations, or AI models.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from core.simulator import blind_spot_tracker

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/blindspots", tags=["BlindSpots"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class BlindSpotRequest(BaseModel):
    target_system: str
    max_checks: Optional[int] = 50


class BlindSpotResult(BaseModel):
    id: str
    type: str
    description: str
    severity: str
    detected_at: Optional[str] = None


class BlindSpotResponse(BaseModel):
    total: int
    results: List[BlindSpotResult]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/analyze", response_model=BlindSpotResponse)
async def analyze_blind_spots(request: BlindSpotRequest) -> BlindSpotResponse:
    """
    Analyze the specified target system for detection blind spots.
    """
    try:
        results = blind_spot_tracker.scan(
            target=request.target_system,
            max_checks=request.max_checks
        )
        response_results = [
            BlindSpotResult(
                id=r["id"],
                type=r["type"],
                description=r["description"],
                severity=r.get("severity", "medium"),
                detected_at=r.get("detected_at"),
            )
            for r in results
        ]
        return BlindSpotResponse(total=len(response_results), results=response_results)
    except Exception as e:
        logger.exception("Blind spot analysis failed")
        raise HTTPException(status_code=500, detail=f"Blind spot analysis error: {e}")


@router.get("/categories", response_model=List[str])
async def list_blind_spot_categories() -> List[str]:
    """
    Return the types/categories of blind spots that can be detected.
    """
    try:
        return blind_spot_tracker.get_categories()
    except Exception as e:
        logger.exception("Failed to retrieve blind spot categories")
        raise HTTPException(status_code=500, detail=f"Could not fetch categories: {e}")
