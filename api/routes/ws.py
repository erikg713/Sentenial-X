import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from ..deps import secure_dep

router = APIRouter(prefix="/ws", tags=["websocket"])

@router.websocket("/telemetry")
async def ws_telemetry(websocket: WebSocket, source: str, filter: str = ""):
    # NOTE: WebSocket cannot enforce Header-based deps easily; you can add token in query if needed.
    await websocket.accept()
    try:
        from cli.telemetry import Telemetry as CLITelemetry
    except Exception as e:
        await websocket.close(code=1011)
        return

    telemetry = CLITelemetry()
    try:
        async for evt in telemetry.stream(source=source, filter_expr=filter):
            await websocket.send_text(json.dumps(evt))
    except WebSocketDisconnect:
        return
    except Exception as e:
        await websocket.send_text(json.dumps({"error": str(e)}))
        await websocket.close(code=1011)
