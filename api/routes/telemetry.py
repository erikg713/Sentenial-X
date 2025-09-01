import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from ..deps import secure_dep
from ..utils import now_iso

router = APIRouter(prefix="/telemetry", tags=["telemetry"])

@router.get("/stream")
async def stream(source: str, filter: str = "", _=Depends(secure_dep)):
    """
    Server-Sent Events (SSE) style stream for live telemetry.
    """
    try:
        from cli.telemetry import Telemetry as CLITelemetry
    except Exception as e:
        raise HTTPException(500, f"Module import failed: {e}")

    telemetry = CLITelemetry()

    async def event_generator():
        async for evt in telemetry.stream(source=source, filter_expr=filter):
            yield f"data: {json.dumps(evt)}\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "Content-Type": "text/event-stream",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_generator(), headers=headers)

@router.get("/heartbeat")
async def heartbeat(_=Depends(secure_dep)):
    return {"status": "alive", "timestamp": now_iso()}
