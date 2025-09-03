"""
WormGPT Models
--------------
Request and response schemas for WormGPT emulation.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any


class WormGPTRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for WormGPT")
    temperature: float = Field(0.7, ge=0.0, le=1.0, description="Sampling temperature")


class WormGPTResponse(BaseModel):
    action: str = Field(..., description="Action inferred from WormGPT")
    prompt: str = Field(..., description="Original prompt")
    prompt_risk: str = Field(..., description="Risk classification of the prompt")
    detections: List[str] = Field(default_factory=list, description="List of triggered detections")
    countermeasures: List[str] = Field(default_factory=list, description="Recommended countermeasures")
    temperature: float = Field(..., description="Temperature used in generation")
    timestamp: str = Field(..., description="Timestamp of generation (ISO format)")
