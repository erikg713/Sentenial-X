from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "Sentenial-X API"

class WormGPTRequest(BaseModel):
    prompt: str
    temperature: float = Field(0.7, ge=0.0, le=1.0)

class WormGPTResponse(BaseModel):
    action: str
    prompt: str
    prompt_risk: str
    detections: List[str]
    countermeasures: List[str]
    temperature: float
    timestamp: str

class CortexRequest(BaseModel):
    source: str
    filter: Optional[str] = None

class TelemetryEvent(BaseModel):
    source: str
    event_type: str
    severity: str
    timestamp: str
    data: Optional[Dict[str, Any]] = None

class OrchestratorRequest(BaseModel):
    action: str
    params: Dict[str, Any] = {}

class OrchestratorResponse(BaseModel):
    action: str
    status: str
    result: Dict[str, Any]

class AlertRequest(BaseModel):
    type: str
    severity: str = "medium"
    payload: Optional[Dict[str, Any]] = None

class AlertResponse(BaseModel):
    id: str
    status: str
    severity: str
    type: str
    timestamp: str
# api/models.py
"""
Pydantic models for Sentenial-X API
Includes request and response schemas for Cortex, WormGPT, Orchestrator, Telemetry, and Exploits
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


# ----------------------------
# Telemetry Models
# ----------------------------
class TelemetryResponse(BaseModel):
    cpu: str
    memory: str
    disk: Optional[str] = None
    network: Optional[str] = None


# ----------------------------
# Orchestrator Models
# ----------------------------
class OrchestratorActionResponse(BaseModel):
    message: str


# ----------------------------
# Cortex Models
# ----------------------------
class ThreatAnalysisRequest(BaseModel):
    threat: Dict[str, Any] = Field(..., description="Threat data to be analyzed")


class ThreatAnalysisResponse(BaseModel):
    threat: Dict[str, Any]
    confidence: float = Field(..., description="Confidence score between 0 and 1")


# ----------------------------
# WormGPT Models
# ----------------------------
class WormGPTRequest(BaseModel):
    payload: Dict[str, Any] = Field(..., description="Payload for WormGPT emulation")


class WormGPTResponse(BaseModel):
    payload: Dict[str, Any]
    status: str = Field(default="emulated")


# ----------------------------
# Exploits Models
# ----------------------------
class ExploitListResponse(BaseModel):
    exploits: List[str]
