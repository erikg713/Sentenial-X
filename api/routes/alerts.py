# -*- coding: utf-8 -*-
"""
Alerts API Routes for Sentenial-X
---------------------------------

Provides endpoints to manage alerts:
- Create alerts
- List & query alerts
- Stream alerts in real-time
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.engine import incident_logger

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/alerts",
    tags=["Alerts"],
    responses={404: {"description": "Not found"}},
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class AlertCreateRequest(BaseModel):
    """Request body for creating an alert manually."""
    severity: str
    source: str
    description: str
    metadata: Optional[dict] = None


class AlertResponse(BaseModel):
    """Response model for a single alert."""
    id: str
    severity: str
    source: str
    description: str
    timestamp: datetime
    metadata: Optional[dict] = None


class AlertListResponse(BaseModel):
    """Response model for listing alerts."""
    alerts: List[AlertResponse]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/", response_model=AlertResponse)
async def create_alert(request: AlertCreateRequest) -> AlertResponse:
    """
    Create a new alert (manual or system-driven).
    """
    try:
        alert = incident_logger.log_alert(
            severity=request.severity,
            source=request.source,
            description=request.description,
            metadata=request.metadata,
        )
        return AlertResponse(**alert)
    except Exception as e:
        logger.exception("Failed to create alert")
        raise HTTPException(status_code=500, detail=f"Alert creation error: {e}")


@router.get("/", response_model=AlertListResponse)
async def list_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity"),
    source: Optional[str] = Query(None, description="Filter by source"),
    limit: int = Query(50, ge=1, le=500, description="Max number of alerts to fetch"),
) -> AlertListResponse:
    """
    List alerts with optional filtering by severity and source.
    """
    try:
        alerts = incident_logger.fetch_alerts(severity=severity, source=source, limit=limit)
        return AlertListResponse(alerts=[AlertResponse(**a) for a in alerts])
    except Exception as e:
        logger.exception("Failed to fetch alerts")
        raise HTTPException(status_code=500, detail=f"Fetch error: {e}")


@router.get("/stream")
async def stream_alerts():
    """
    Stream alerts in real-time using Server-Sent Events (SSE).
    """
    try:
        async def event_generator():
            async for alert in incident_logger.stream_alerts():
                yield f"data: {alert}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except Exception as e:
        logger.exception("Failed to stream alerts")
        raise HTTPException(status_code=500, detail=f"Stream error: {e}")
