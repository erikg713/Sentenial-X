from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from api.controllers.threats_controller import ThreatsController
from api.utils.auth import get_current_user

router = APIRouter(tags=["Threats"])
controller = ThreatsController()


# ---------- Request / Response Schemas ---------- #

class ThreatSimulationPayload(BaseModel):
    indicator: str = Field(..., description="The threat indicator to simulate (e.g., IP, domain, hash)")
    severity: str = Field(..., description="Severity level", regex="^(low|medium|high)$")
    tags: Optional[list[str]] = Field(default_factory=list, description="List of related tags or TTPs")
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict, description="Additional contextual metadata")


class ThreatResponse(BaseModel):
    id: str
    indicator: str
    severity: str
    risk_score: int
    tags: list[str]
    metadata: dict[str, Any]


# ---------- Routes ---------- #

@router.get("/", response_model=List[ThreatResponse], status_code=status.HTTP_200_OK)
async def list_threats(user: dict = Depends(get_current_user)):
    """
    Retrieve all tracked threats for the authenticated user.
    """
    return await controller.list_threats()


@router.post("/simulate", response_model=ThreatResponse, status_code=status.HTTP_201_CREATED)
async def simulate_threat(payload: ThreatSimulationPayload, user: dict = Depends(get_current_user)):
    """
    Simulate a threat scenario for training/testing purposes.
    This does NOT represent a live attack; it's for validating detection rules and playbooks.
    """
    result = await controller.simulate_threat(payload.dict())
    if not result:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Threat simulation failed")
    return result
