"""
Semantic Analyzer REST API Server
=================================

Provides HTTP endpoints for analyzing telemetry/events using semantic analysis.
Runs as a FastAPI app with health checks, structured logging, and async support.
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from .analyzer import analyze_event
from . import logger


# ------------------------------
# Request/Response Models
# ------------------------------
class TelemetryEvent(BaseModel):
    event: Dict[str, Any]


class AnalysisResult(BaseModel):
    score: float
    tags: list[str]
    explanation: str


# ------------------------------
# FastAPI App Setup
# ------------------------------
app = FastAPI(
    title="Sentenial-X Semantic Analyzer",
    description="API service for semantic event analysis and stealth scoring.",
    version="1.0.0",
)


# ------------------------------
# Routes
# ------------------------------
@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok", "service": "semantic_analyzer"}


@app.post("/analyze", response_model=AnalysisResult)
async def analyze(event: TelemetryEvent):
    """
    Analyze a telemetry event using semantic analysis engine.
    """
    try:
        logger.info(f"Received event for analysis: {event.event}")
        result = analyze_event(event.event)
        return AnalysisResult(
            score=result.get("score", 0.0),
            tags=result.get("tags", []),
            explanation=result.get("explanation", ""),
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Analysis error")


# ------------------------------
# Entrypoint
# ------------------------------
def run_server(host: str = "0.0.0.0", port: int = 8082):
    """Run the FastAPI server with Uvicorn."""
    logger.info(f"Starting Semantic Analyzer API on {host}:{port}")
    uvicorn.run("core.semantic_analyzer.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run_server()
