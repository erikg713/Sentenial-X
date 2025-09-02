# apps/dashboard/widgets/telemetry_chart.py

"""
Telemetry Chart Widget
----------------------
Provides a live telemetry data feed for visualization on the dashboard.
Includes CPU, memory, disk, and network usage metrics.
"""

from fastapi import APIRouter
from typing import Dict
from agents.telemetry import TelemetryAgent

router = APIRouter()
telemetry_agent = TelemetryAgent()

@router.get("/data")
async def telemetry_data() -> Dict[str, Dict]:
    """
    Retrieve latest telemetry metrics from the system.
    """
    metrics = telemetry_agent.get_metrics()
    return {
        "status": "success",
        "message": "Telemetry data fetched",
        "data": metrics
    }

# Example usage:
# GET /widgets/telemetry_chart/data
