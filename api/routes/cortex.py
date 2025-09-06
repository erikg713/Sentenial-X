# -*- coding: utf-8 -*-
"""
Cortex API Routes for Sentenial-X
---------------------------------

Provides endpoints for:
- Running semantic log analysis
- Extracting threat intelligence
- NLP-based anomaly detection
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from core.cortex import semantic_analyzer

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/cortex",
    tags=["Cortex"],
    responses={404: {"description": "Not found"}},
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class CortexAnalyzeRequest(BaseModel):
    """Request body for semantic log analysis."""
    source: str
    filter: Optional[str] = None
    max_results: int = 100


class CortexInsight(BaseModel):
    """Single insight returned by the analyzer."""
    type: str
    message: str
    confidence: float
    metadata: Optional[dict] = None


class CortexResponse(BaseModel):
    """Response containing multiple insights."""
    insights: List[CortexInsight]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/analyze", response_model=CortexResponse)
async def analyze_logs(request: CortexAnalyzeRequest) -> CortexResponse:
    """
    Run semantic log analysis through the Cortex engine.
    """
    try:
        results = semantic_analyzer.analyze_logs(
            source=request.source,
            filter_pattern=request.filter,
            max_results=request.max_results,
        )
        insights = [
            CortexInsight(
                type=r["type"],
                message=r["message"],
                confidence=r.get("confidence", 0.0),
                metadata=r.get("metadata"),
            )
            for r in results
        ]
        return CortexResponse(insights=insights)
    except Exception as e:
        logger.exception("Cortex analysis failed")
        raise HTTPException(status_code=500, detail=f"Cortex analysis error: {e}")


@router.get("/intelligence", response_model=CortexResponse)
async def extract_intelligence(
    keyword: str = Query(..., description="Keyword to search for"),
    max_results: int = Query(50, ge=1, le=500, description="Number of insights to fetch"),
) -> CortexResponse:
    """
    Extract structured threat intelligence by keyword.
    """
    try:
        results = semantic_analyzer.extract_intelligence(keyword=keyword, max_results=max_results)
        insights = [
            CortexInsight(
                type=r["type"],
                message=r["message"],
                confidence=r.get("confidence", 0.0),
                metadata=r.get("metadata"),
            )
            for r in results
        ]
        return CortexResponse(insights=insights)
    except Exception as e:
        logger.exception("Failed to extract threat intelligence")
        raise HTTPException(status_code=500, detail=f"Threat intelligence error: {e}")
