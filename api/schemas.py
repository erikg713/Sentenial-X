# api/schemas.py
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
import datetime

# ----------------------------
# Health / System / Utilities
# ----------------------------
class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "Sentenial-X API"

class SystemInfoResponse(BaseModel):
    os: str
    os_version: str
    cpu_percent: float
    memory_total: int
    memory_used: int
    memory_percent: float
    disk_total: int
    disk_used: int
    disk_percent: float
    net_bytes_sent: int
    net_bytes_recv: int
    timestamp: datetime.datetime

# ----------------------------
# WormGPT
# ----------------------------
class WormGPTRequest(BaseModel):
    payload: Dict[str, Any]

class WormGPTResponse(BaseModel):
    payload: Dict[str, Any]
    status: str = Field(default="emulated")

# ----------------------------
# Cortex
# ----------------------------
class ThreatAnalysisRequest(BaseModel):
    threat: Dict[str, Any] = Field(..., description="Threat data to be analyzed")

class ThreatAnalysisResponse(BaseModel):
    threat: Dict[str, Any]
    confidence: float = Field(..., description="Confidence score 0-1")

# ----------------------------
# Telemetry
# ----------------------------
class TelemetryEvent(BaseModel):
    source: str
    event_type: str
    severity: str
    timestamp: str
    data: Optional[Dict[str, Any]] = None

class TelemetryResponse(BaseModel):
    cpu: str
    memory: str
    disk: Optional[str] = None
    network: Optional[str] = None

# ----------------------------
# Orchestrator
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
# Alerts
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
# Exploits
# ----------------------------
class ExploitListResponse(BaseModel):
    exploits: List[str]
