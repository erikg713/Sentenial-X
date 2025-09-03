"""
Alert Models
------------
Schemas for alerts within the Semantic Analyzer.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class AlertRequest(BaseModel):
    type: str = Field(..., description="Alert type")
    severity: str = Field(default="medium", description="Severity level (low, medium, high, critical)")
    payload: Optional[Dict[str, Any]] = Field(default=None, description="Optional payload data")


class AlertResponse(BaseModel):
    id: str = Field(..., description="Alert ID")
    status: str = Field(..., description="Processing status")
    severity: str = Field(..., description="Severity level")
    type: str = Field(..., description="Alert type")
    timestamp: str = Field(..., description="Timestamp (ISO format)")
