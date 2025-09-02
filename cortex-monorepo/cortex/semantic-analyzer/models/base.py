from pydantic import BaseModel, Field
from typing import Dict, Any

class SemanticEvent(BaseModel):
    """
    Base model for any event processed by the semantic analyzer.
    """
    event_id: str
    source: str
    timestamp: str
    raw_data: Dict[str, Any]

class AnalysisResult(BaseModel):
    """
    Standardized AI analysis output.
    """
    event_id: str
    severity: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    summary: str
    recommendations: Dict[str, Any]
