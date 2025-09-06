# -*- coding: utf-8 -*-
"""
WormGPT Detection API Routes for Sentenial-X
--------------------------------------------

Exposes endpoints to detect, analyze, and report adversarial AI prompts (WormGPT).
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from core.simulator import wormgpt_clone

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/wormgpt", tags=["WormGPT"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class WormGPTAnalyzeRequest(BaseModel):
    prompt: str
    max_depth: Optional[int] = 3


class WormGPTInsight(BaseModel):
    prompt: str
    risk_score: float
    categories: List[str]
    details: Optional[dict] = None


class WormGPTResponse(BaseModel):
    insights: List[WormGPTInsight]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/analyze", response_model=WormGPTResponse)
async def analyze_prompt(request: WormGPTAnalyzeRequest) -> WormGPTResponse:
    """
    Analyze a prompt to detect potential WormGPT malicious/rogue patterns.
    """
    try:
        results = wormgpt_clone.analyze_prompt(prompt=request.prompt, max_depth=request.max_depth)
        insights = [
            WormGPTInsight(
                prompt=r["prompt"],
                risk_score=r["risk_score"],
                categories=r.get("categories", []),
                details=r.get("details"),
            )
            for r in results
        ]
        return WormGPTResponse(insights=insights)
    except Exception as e:
        logger.exception("WormGPT analysis failed")
        raise HTTPException(status_code=500, detail=f"WormGPT analysis error: {e}")


@router.get("/categories", response_model=List[str])
async def list_categories() -> List[str]:
    """
    Return the list of categories that the WormGPT analyzer can detect.
    """
    try:
        return wormgpt_clone.get_categories()
    except Exception as e:
        logger.exception("Failed to retrieve WormGPT categories")
        raise HTTPException(status_code=500, detail=f"Could not fetch categories: {e}")
