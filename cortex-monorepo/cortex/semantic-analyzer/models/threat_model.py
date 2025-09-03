"""
Threat Model
------------
Defines the schema for representing threats analyzed by the Semantic Analyzer.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any


class Threat(BaseModel):
    id: str = Field(..., description="Unique identifier for the threat")
    type: str = Field(..., description="Threat type (e.g., malware, phishing, rce)")
    severity: str = Field(..., description="Severity level (low, medium, high, critical)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context data")
    detected_at: str = Field(..., description="Timestamp of detection (ISO format)")
