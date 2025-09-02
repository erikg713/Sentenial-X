from fastapi import APIRouter, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
from typing import List, Dict, Any
from api.utils.logger import init_logger
from api.utils.auth import verify_api_key
from cortex.semantic_analyzer.pipeline import SemanticAnalyzerPipeline

logger = init_logger("semantic_analyzer_api")
router = APIRouter()
pipeline = SemanticAnalyzerPipeline()

# API Key header dependency
api_key_header = APIKeyHeader(name="x-api-key", auto_error=True)


# -------------------------------
# Request / Response Models
# -------------------------------
from pydantic import BaseModel, Field

class RawLogItem(BaseModel):
    log: str = Field(..., description="Raw log or alert string")
    source: str = Field(..., description="Origin of the log/event")


class BatchLogsRequest(BaseModel):
    logs: List[RawLogItem]


class AnalysisResponseItem(BaseModel):
    event_id: str
    source: str
    timestamp: str
    analysis: str
    severity: str
    risk_score: float
    countermeasures: List[str]


class BatchAnalysisResponse(BaseModel):
    results: List[AnalysisResponseItem]


# -------------------------------
# API Endpoint
# -------------------------------
@router.post(
    "/analyze",
    response_model=BatchAnalysisResponse,
    summary="Run semantic analysis on raw logs or alerts",
)
async def analyze_logs(
    payload: BatchLogsRequest,
    x_api_key: str = Depends(api_key_header),
):
    # Verify API Key
    verify_api_key(x_api_key)

    try:
        # Transform input for pipeline
        raw_logs = [{"log": item.log, "source": item.source} for item in payload.logs]
        logger.info(f"Received {len(raw_logs)} logs for analysis")

        # Run async semantic analyzer pipeline
        results: List[Dict[str, Any]] = await pipeline.process_batch(raw_logs)

        logger.info("Semantic analysis completed successfully")
        return {"results": results}

    except Exception as e:
        logger.error(f"Semantic analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
