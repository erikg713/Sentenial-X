from fastapi import APIRouter, Depends
from ..models import HealthResponse
from ..deps import secure_dep

router = APIRouter(prefix="/health", tags=["health"])

@router.get("", response_model=HealthResponse)
async def health(_=Depends(secure_dep)):
    return HealthResponse()
