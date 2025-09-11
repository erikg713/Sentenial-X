"""
Semantic Analyzer REST API Server
=================================

Provides HTTP endpoints for analyzing telemetry/events using semantic analysis.

Improvements made:
- better type validation and pydantic Field usage
- async-safe invocation of analyze_event (supports coroutine and sync functions)
- request timing middleware with structured logs
- lightweight in-memory metrics and /metrics endpoint
- startup/shutdown hooks and uptime in /health
- clearer error handling and stricter return validation
- CLI args and env override for host/port when run as __main__
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import os
import time
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .analyzer import analyze_event
from . import logger

# ------------------------------
# Request/Response Models
# ------------------------------
class TelemetryEvent(BaseModel):
    event: Dict[str, Any] = Field(..., description="Raw telemetry/event payload to analyze")


class AnalysisResult(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized stealth/confidence score (0.0 - 1.0)")
    tags: List[str] = Field(default_factory=list, description="List of classification tags")
    explanation: Optional[str] = Field("", description="Short explanation / reasoning for the score")


# ------------------------------
# FastAPI App Setup
# ------------------------------
app = FastAPI(
    title="Sentenial-X Semantic Analyzer",
    description="API service for semantic event analysis and stealth scoring.",
    version="1.1.0",
)

# Simple in-memory metrics (kept intentionally lightweight; replace with Prometheus client if needed)
_metrics = {
    "total_requests": 0,
    "total_errors": 0,
    "last_request_timestamp": 0.0,
}
_metrics_lock = asyncio.Lock()


# ------------------------------
# Lifecycle Events
# ------------------------------
@app.on_event("startup")
async def _startup():
    """Record start time and log startup."""
    app.state.start_time = time.time()
    logger.info("Semantic Analyzer starting up")


@app.on_event("shutdown")
async def _shutdown():
    """Log shutdown."""
    logger.info("Semantic Analyzer shutting down")


# ------------------------------
# Middleware
# ------------------------------
@app.middleware("http")
async def _metrics_middleware(request: Request, call_next):
    """Measure request duration, increment metrics, and log request info."""
    start = time.time()
    try:
        response = await call_next(request)
        return response
    finally:
        elapsed = time.time() - start
        # best-effort metrics update
        try:
            async with _metrics_lock:
                _metrics["total_requests"] += 1
                _metrics["last_request_timestamp"] = time.time()
        except Exception:
            # Do not fail the request because metrics update failed
            logger.warning("Metrics update failed", exc_info=True)
        logger.info(
            "%s %s %d %.3fs",
            request.method,
            request.url.path,
            getattr(response, "status_code", 0),
            elapsed,
        )


# ------------------------------
# Routes
# ------------------------------
@app.get("/health")
async def health_check():
    """Basic health check endpoint with uptime and basic metrics."""
    uptime = time.time() - getattr(app.state, "start_time", time.time())
    # produce a concise health response
    async with _metrics_lock:
        metrics_snapshot = dict(_metrics)
    return {
        "status": "ok",
        "service": "semantic_analyzer",
        "uptime_seconds": round(uptime, 3),
        "metrics": metrics_snapshot,
    }


@app.get("/metrics")
async def metrics_endpoint():
    """
    Lightweight plaintext metrics endpoint. Format is inspired by Prometheus
    but kept intentionally simple to avoid adding dependencies.
    """
    async with _metrics_lock:
        total_requests = _metrics["total_requests"]
        total_errors = _metrics["total_errors"]
        last_ts = _metrics["last_request_timestamp"]

    body_lines = [
        f"# TYPE semantic_analyzer_total_requests counter",
        f"semantic_analyzer_total_requests {total_requests}",
        f"# TYPE semantic_analyzer_total_errors counter",
        f"semantic_analyzer_total_errors {total_errors}",
        f"# TYPE semantic_analyzer_last_request_timestamp gauge",
        f"semantic_analyzer_last_request_timestamp {int(last_ts)}",
    ]
    return PlainTextResponse("\n".join(body_lines), media_type="text/plain; charset=utf-8")


@app.post("/analyze", response_model=AnalysisResult, status_code=status.HTTP_200_OK)
async def analyze(event: TelemetryEvent):
    """
    Analyze a telemetry event using the semantic analysis engine.

    - Supports analyzer functions that are either synchronous or async.
    - Ensures the analyzer returns a dict with expected keys.
    - Protects the server from analyzer blocking by running sync analyzers in a thread pool.
    """
    logger.info("Received event for analysis", extra={"event_preview": _preview_event(event.event)})
    try:
        # call analyze_event in an async-safe manner: await if coroutine, otherwise run in executor
        if inspect.iscoroutinefunction(analyze_event):
            result = await analyze_event(event.event)
        else:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, analyze_event, event.event)

        # Validate result shape
        if not isinstance(result, dict):
            raise TypeError("analyze_event must return a dict-like result")

        # Extract fields with safe defaults and types
        score = float(result.get("score", 0.0))
        tags = list(result.get("tags", []))
        explanation = result.get("explanation", "")
        if explanation is None:
            explanation = ""

        # update metrics
        async with _metrics_lock:
            # total_requests incremented by middleware; only increment errors here
            pass

        return AnalysisResult(score=score, tags=tags, explanation=str(explanation))

    except HTTPException:
        # Re-raise FastAPI HTTPExceptions without modification
        raise
    except Exception as exc:
        # Log full exception details server-side, but return a sanitized message to the client
        logger.exception("Analysis failed")
        async with _metrics_lock:
            _metrics["total_errors"] += 1
        raise HTTPException(status_code=500, detail="Internal analysis error")


# ------------------------------
# Helpers
# ------------------------------
def _preview_event(evt: Dict[str, Any], max_len: int = 256) -> str:
    """
    Create a small preview of the event for logs without dumping huge payloads.
    Keeps logs informative but avoids sensitive/big dumps.
    """
    try:
        s = str(evt)
        if len(s) <= max_len:
            return s
        return s[: max_len - 3] + "..."
    except Exception:
        return "<unserializable event>"


# ------------------------------
# Entrypoint
# ------------------------------
def run_server(host: str = "0.0.0.0", port: int = 8082):
    """Run the FastAPI server with Uvicorn."""
    logger.info(f"Starting Semantic Analyzer API on {host}:{port}")
    # Allow environment to override log level via UVICORN_LOG_LEVEL or use info
    uvicorn.run(app, host=host, port=port, log_level=os.getenv("UVICORN_LOG_LEVEL", "info"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="semantic_analyzer", description="Run the Sentenial-X semantic analyzer API.")
    parser.add_argument("--host", "-H", default=os.getenv("SA_HOST", "0.0.0.0"), help="Host to bind to")
    parser.add_argument("--port", "-P", type=int, default=int(os.getenv("SA_PORT", "8082")), help="Port to listen on")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
