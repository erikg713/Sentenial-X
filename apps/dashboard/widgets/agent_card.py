# apps/dashboard/widgets/agent_card.py

"""
Agent Card Widget
-----------------
Displays the status, health, and metrics of a single agent within the
Sentenial-X Dashboard.
"""

from fastapi import APIRouter
from api.utils.response import success_response
from agents.telemetry import TelemetryAgent

router = APIRouter()
telemetry_agent = TelemetryAgent()

@router.get("/status/{agent_id}")
async def agent_status(agent_id: str):
    """
    Fetch the current status and telemetry of a specific agent.
    """
    status = telemetry_agent.get_agent_status(agent_id)
    return success_response(f"Agent {agent_id} status fetched", status)

@router.get("/metrics/{agent_id}")
async def agent_metrics(agent_id: str):
    """
    Fetch the latest telemetry metrics for a specific agent.
    """
    metrics = telemetry_agent.get_agent_metrics(agent_id)
    return success_response(f"Agent {agent_id} metrics fetched", metrics)

# Example usage:
# GET /widgets/agent_card/status/agent-001
# GET /widgets/agent_card/metrics/agent-001
