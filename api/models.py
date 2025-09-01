# api/models.py
"""
Pydantic models for Sentenial-X API
Includes request and response schemas for Cortex, WormGPT, Orchestrator, Telemetry, Exploits, and Alerts
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

# ----------------------------
# Health / Status Models
# ----------------------------
class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "Sentenial-X API"


# ----------------------------
# WormGPT Models
# ----------------------------
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

class WormGPTPayloadRequest(BaseModel):
    payload: Dict[str, Any] = Field(..., description="Payload for WormGPT emulation")

class WormGPTPayloadResponse(BaseModel):
    payload: Dict[str, Any]
    status: str = Field(default="emulated")


# ----------------------------
# Cortex Models
# ----------------------------
class CortexRequest(BaseModel):
    source: str
    filter: Optional[str] = None

class ThreatAnalysisRequest(BaseModel):
    threat: Dict[str, Any] = Field(..., description="Threat data to be analyzed")

class ThreatAnalysisResponse(BaseModel):
    threat: Dict[str, Any]
    confidence: float = Field(..., description="Confidence score between 0 and 1")


# ----------------------------
# Telemetry Models
# ----------------------------
class TelemetryResponse(BaseModel):
    cpu: str
    memory: str
    disk: Optional[str] = None
    network: Optional[str] = None

class TelemetryEvent(BaseModel):
    source: str
    event_type: str
    severity: str
    timestamp: str
    data: Optional[Dict[str, Any]] = None


# ----------------------------
# Orchestrator Models
# ----------------------------
class OrchestratorRequest(BaseModel):
    action: str
    params: Dict[str, Any] = {}

class OrchestratorResponse(BaseModel):
    action: str
    status: str
    result: Dict[str, Any]

class OrchestratorActionResponse(BaseModel):
    message: str


# ----------------------------
# Alerts Models
# ----------------------------
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


# ----------------------------
# Exploits Models
# ----------------------------
class ExploitListResponse(BaseModel):
    exploits: List[str]
