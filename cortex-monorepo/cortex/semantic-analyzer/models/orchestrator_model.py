"""
Orchestrator Models
-------------------
Schemas for orchestrator requests and responses.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any


class OrchestratorRequest(BaseModel):
    action: str = Field(..., description="Action to be orchestrated")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the action")


class OrchestratorResponse(BaseModel):
    action: str = Field(..., description="Action that was executed")
    status: str = Field(..., description="Execution status (success/failure)")
    result: Dict[str, Any] = Field(default_factory=dict, description="Result of the orchestration")
