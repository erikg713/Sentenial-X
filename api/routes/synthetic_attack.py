# -*- coding: utf-8 -*-
"""
Synthetic Attack Fuzzer API Routes for Sentenial-X
--------------------------------------------------

Exposes endpoints to perform synthetic attack fuzzing on target systems,
generating test payloads and identifying potential vulnerabilities.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from core.simulator import synthetic_attack_fuzzer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/synthetic_attack", tags=["SyntheticAttack"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class FuzzerRequest(BaseModel):
    target: str
    payload_count: Optional[int] = 50
    max_depth: Optional[int] = 3


class FuzzerResult(BaseModel):
    id: str
    payload: str
    outcome: str
    severity: str
    detected_at: Optional[str] = None


class FuzzerResponse(BaseModel):
    total: int
    results: List[FuzzerResult]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/run", response_model=FuzzerResponse)
async def run_synthetic_attack(request: FuzzerRequest) -> FuzzerResponse:
    """
    Run synthetic attack fuzzing against a target system.
    """
    try:
        results = synthetic_attack_fuzzer.fuzz(
            target=request.target,
            payload_count=request.payload_count,
            max_depth=request.max_depth
        )
        response_results = [
            FuzzerResult(
                id=r["id"],
                payload=r["payload"],
                outcome=r.get("outcome", "unknown"),
                severity=r.get("severity", "medium"),
                detected_at=r.get("detected_at"),
            )
            for r in results
        ]
        return FuzzerResponse(total=len(response_results), results=response_results)
    except Exception as e:
        logger.exception("Synthetic attack fuzzing failed")
        raise HTTPException(status_code=500, detail=f"Fuzzer error: {e}")


@router.get("/categories", response_model=List[str])
async def list_fuzzer_categories() -> List[str]:
    """
    Return the types of attacks or payload categories the fuzzer can generate.
    """
    try:
        return synthetic_attack_fuzzer.get_categories()
    except Exception as e:
        logger.exception("Failed to retrieve fuzzer categories")
        raise HTTPException(status_code=500, detail=f"Could not fetch categories: {e}")
