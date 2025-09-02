from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any
from api.utils.auth import verify_api_key
from services.threat_engine.detectors.ML.infer import MLInfer
import logging

logger = logging.getLogger("sentenialx.api.ml_detector")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

router = APIRouter()

# Initialize ML Inference
ml_infer = MLInfer()


class MLRequest(BaseModel):
    data: Dict[str, Any]


class MLResponse(BaseModel):
    predicted_class: int
    confidence: float


@router.post("/predict", response_model=MLResponse, dependencies=[Depends(verify_api_key)])
async def predict_threat(request: MLRequest):
    """
    Predict the threat type using ML detector.
    """
    try:
        result = ml_infer.predict(request.data)
        return MLResponse(**result)
    except Exception as e:
        logger.error("ML prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"ML prediction failed: {str(e)}")
