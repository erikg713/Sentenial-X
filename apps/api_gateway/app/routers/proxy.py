# app/routers/proxy.py
from fastapi import APIRouter, Depends, HTTPException
from ..deps import require_scope
from ..config import settings
import httpx

router = APIRouter(prefix="/proxy", tags=["proxy"])


@router.get("/threats")
async def proxy_threats(payload=Depends(require_scope("read:threats"))):
    """
    Example proxy endpoint to call Threat Engine.
    Requires: READ scope for threats.
    """
    if not settings.THREAT_ENGINE_URL:
        raise HTTPException(status_code=503, detail="Threat engine not configured")

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{settings.THREAT_ENGINE_URL}/v1/alerts")
        resp.raise_for_status()
        return resp.json()