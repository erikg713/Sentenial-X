# app/routers/health.py
from fastapi import APIRouter
from ..models import HealthResponse
import time

router = APIRouter(prefix="/health", tags=["health"])
START_TS = time.time()


@router.get("/", response_model=HealthResponse, summary="Service health")
async def health():
    uptime = int(time.time() - START_TS)
    return HealthResponse(service="api-gateway", status="ok", uptime_seconds=uptime)