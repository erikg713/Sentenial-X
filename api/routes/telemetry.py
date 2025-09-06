# -*- coding: utf-8 -*-
"""
Telemetry API Routes for Sentenial-X
------------------------------------

Exposes telemetry collection, summaries, and query endpoints.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from core.simulator import TelemetryCollector

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/telemetry", tags=["Telemetry"])

collector = TelemetryCollector()


class TelemetryResponse(BaseModel):
    count: int
    data: list


@router.get("/summary", response_model=dict)
async def telemetry_summary() -> dict:
    """Get telemetry summary."""
    return collector.summary()


@router.get("/report", response_model=TelemetryResponse)
async def telemetry_report(last_n: Optional[int] = Query(None, description="Last N entries")) -> TelemetryResponse:
    """Get detailed telemetry report."""
    data = collector.report(last_n=last_n)
    return TelemetryResponse(count=len(data), data=data)
