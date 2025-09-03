"""
Telemetry Model
---------------
Represents system or network telemetry events.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class TelemetryEvent(BaseModel):
    source: str = Field(..., description="Telemetry data source")
    metric: str = Field(..., description="Name of the telemetry metric")
    value: str = Field(..., description="Metric value")
    timestamp: str = Field(..., description="Timestamp (ISO format)")
    tags: Optional[Dict[str, Any]] = Field(default=None, description="Optional tags or metadata")
