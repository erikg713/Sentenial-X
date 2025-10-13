# apps/api-gateway/app/routers/cortex_routes.py
from fastapi import APIRouter, Depends
import requests
from app.deps import get_current_user  # RBAC dependency

router = APIRouter(prefix="/cortex", tags=["cortex"])

CORTEX_SERVICE_URL = "http://cortex-service:8080"  # Docker service name

@router.post("/predict")
def proxy_predict(data: dict, user=Depends(get_current_user)):
    # Policy check placeholder (integrate with policy engine)
    if not user.has_permission("analyze_logs"):
        raise HTTPException(403, "Unauthorized")
    
    response = requests.post(f"{CORTEX_SERVICE_URL}/predict", json=data)
    response.raise_for_status()
    return response.json()
