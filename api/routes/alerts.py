from fastapi import APIRouter, Depends, HTTPException
from ..models import AlertRequest, AlertResponse
from ..deps import secure_dep

router = APIRouter(prefix="/alerts", tags=["alerts"])

@router.post("/dispatch", response_model=AlertResponse)
async def dispatch(req: AlertRequest, _=Depends(secure_dep)):
    try:
        from cli.alerts import AlertDispatcher
    except Exception as e:
        raise HTTPException(500, f"Module import failed: {e}")

    dispatcher = AlertDispatcher()
    result = await dispatcher.dispatch(alert_type=req.type, severity=req.severity, payload=req.payload or {})
    return AlertResponse(**result)
