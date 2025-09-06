# -*- coding: utf-8 -*-
"""
Sentenial-X Cortex API
---------------------

Provides endpoints for NLP-based threat analysis and semantic detection.

Features:
- Analyze logs or text for potential threats
- Filter results by severity or type
- Integrate results with telemetry
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict
from datetime import datetime

from api.utils import api_response, api_exception, log_api_call
from core.simulator import TelemetryCollector
from core.simulator.cortex_engine import CortexEngine  # assume exists

# ---------------------------------------------------------------------------
# Router setup
# ---------------------------------------------------------------------------
router = APIRouter(
    prefix="/cortex",
    tags=["cortex"]
)

# ---------------------------------------------------------------------------
# Core engine and telemetry
# ---------------------------------------------------------------------------
telemetry_collector = TelemetryCollector()
cortex_engine = CortexEngine()

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@router.get("/")
async def root():
    """
    Simple health check for Cortex API
    """
    return api_response({"status": "Cortex API online"}, message="Cortex endpoint reachable")


@router.post("/analyze")
@log_api_call
def analyze_text(
    text: str = Query(..., description="Text or logs to analyze"),
    severity_filter: Optional[str] = Query(None, description="Optional severity filter: low, medium, high, critical"),
    max_results: int = Query(10, description="Maximum number of results to return")
):
    """
    Perform NLP threat analysis on input text
    """
    if not text.strip():
        api_exception(400, "Input text cannot be empty")

    # Run analysis using CortexEngine
    try:
        results = cortex_engine.analyze(text)
    except Exception as e:
        api_exception(500, f"Cortex analysis failed: {str(e)}")

    # Filter by severity if requested
    if severity_filter:
        severity_filter = severity_filter.lower()
        results = [r for r in results if r.get("severity") == severity_filter]

    # Limit results
    results = results[:max_results]

    # Log to telemetry
    for r in results:
        telemetry_collector.add({
            "event": "cortex_analysis",
            "severity": r.get("severity"),
            "type": r.get("type"),
            "timestamp": datetime.utcnow().isoformat()
        })

    return api_response(results, message=f"{len(results)} analysis results returned")
