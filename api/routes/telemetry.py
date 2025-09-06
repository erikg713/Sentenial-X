# -*- coding: utf-8 -*-
"""
Telemetry API Routes for Sentenial-X
------------------------------------

Exposes endpoints to collect, query, and summarize telemetry
from simulators, attack playbooks, and detection modules.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from core.simulator import TelemetryCollector

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/telemetry", tags=["Telemetry"])

collector = TelemetryCollector()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TelemetryEntry(BaseModel):
    id: str
    source: str
    type: str
    severity: str
    timestamp: datetime
    details: Optional[dict] = None


class TelemetryResponse(BaseModel):
    total: int
    entries: List[TelemetryEntry]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/add", response_model=dict)
async def add_telemetry(
    source: str,
    type: str,
    severity: str = "medium",
    details: Optional[dict] = None
) -> dict:
    """
    Add a telemetry entry to the collector.
    """
    try:
        entry_id = collector.add({
            "source": source,
            "type": type,
            "severity": severity,
            "details": details,
            "timestamp": datetime.utcnow()
        })
        return {"status": "added", "id": entry_id}
    except Exception as e:
        logger.exception("Failed to add telemetry")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary", response_model=dict)
async def summary(last_n: Optional[int] = None) -> dict:
    """
    Return a summary of collected telemetry.
    """
    try:
        return collector.summary(last_n=last_n)
    except Exception as e:
        logger.exception("Failed to retrieve telemetry summary")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=TelemetryResponse)
async def list_entries(last_n: Optional[int] = Query(None, description="Limit number of entries returned")) -> TelemetryResponse:
    """
    List telemetry entries, optionally limited to the last N entries.
    """
    try:
        entries = collector.report(last_n=last_n)
        response_entries = [
            TelemetryEntry(
                id=e["id"],
                source=e["source"],
                type=e["type"],
                severity=e["severity"],
                timestamp=e["timestamp"],
                details=e.get("details")
            )
            for e in entries
        ]
        return TelemetryResponse(total=len(response_entries), entries=response_entries)
    except Exception as e:
        logger.exception("Failed to list telemetry entries")
        raise HTTPException(status_code=500, detail=str(e))
