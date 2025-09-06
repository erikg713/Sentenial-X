# -*- coding: utf-8 -*-
"""
Sentenial-X Alerts API
----------------------

Provides endpoints for managing alerts in the Sentenial-X system.

Features:
- Create new alerts
- List active alerts
- Filter by severity or type
- Mark alerts as resolved
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict
from uuid import uuid4
from datetime import datetime

from api.utils import api_response, api_exception, log_api_call
from core.simulator import TelemetryCollector

# ---------------------------------------------------------------------------
# Router setup
# ---------------------------------------------------------------------------
router = APIRouter(
    prefix="/alerts",
    tags=["alerts"]
)

# ---------------------------------------------------------------------------
# Core telemetry/alert storage
# ---------------------------------------------------------------------------
telemetry_collector = TelemetryCollector()
ALERTS_DB: List[Dict] = []

# ---------------------------------------------------------------------------
# Models (simple dict-based, production-ready can switch to DB)
# ---------------------------------------------------------------------------
def create_alert(alert_type: str, severity: str, message: str) -> Dict:
    alert = {
        "id": str(uuid4()),
        "type": alert_type,
        "severity": severity,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
        "resolved": False
    }
    ALERTS_DB.append(alert)
    telemetry_collector.add({"alert_id": alert["id"], "type": alert_type, "severity": severity})
    return alert

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@router.post("/")
@log_api_call
def post_alert(
    alert_type: str = Query(..., description="Type of alert, e.g., 'malware', 'network'"),
    severity: str = Query(..., description="Severity level: low, medium, high, critical"),
    message: str = Query(..., description="Alert message")
):
    """
    Create a new alert
    """
    if severity.lower() not in {"low", "medium", "high", "critical"}:
        api_exception(400, f"Invalid severity level: {severity}")

    alert = create_alert(alert_type, severity.lower(), message)
    return api_response(alert, message="Alert created successfully")


@router.get("/")
@log_api_call
def list_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity"),
    resolved: Optional[bool] = Query(None, description="Filter by resolved status")
):
    """
    List all alerts, optionally filtered
    """
    filtered = ALERTS_DB
    if severity:
        filtered = [a for a in filtered if a["severity"] == severity.lower()]
    if resolved is not None:
        filtered = [a for a in filtered if a["resolved"] == resolved]

    return api_response(filtered, message=f"{len(filtered)} alerts found")


@router.post("/{alert_id}/resolve")
@log_api_call
def resolve_alert(alert_id: str):
    """
    Mark an alert as resolved
    """
    alert = next((a for a in ALERTS_DB if a["id"] == alert_id), None)
    if not alert:
        api_exception(404, f"Alert with ID {alert_id} not found")

    if alert["resolved"]:
        return api_response(alert, message="Alert already resolved")

    alert["resolved"] = True
    return api_response(alert, message="Alert marked as resolved")
