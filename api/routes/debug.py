# -*- coding: utf-8 -*-
"""
Debug API Routes for Sentenial-X
--------------------------------

Provides endpoints for testing, diagnostics, and internal
debugging purposes. These endpoints are intended for
development and troubleshooting only.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/debug", tags=["Debug"])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/ping")
async def ping() -> Dict[str, Any]:
    """
    Simple ping endpoint for health check.
    """
    logger.debug("Ping received")
    return {"status": "ok", "timestamp": datetime.utcnow()}


@router.get("/echo")
async def echo(message: str) -> Dict[str, Any]:
    """
    Echo back any provided message for testing.
    """
    logger.debug("Echo received: %s", message)
    return {"echo": message, "timestamp": datetime.utcnow()}


@router.get("/config")
async def show_config() -> Dict[str, Any]:
    """
    Return current runtime debug configuration (mock example).
    """
    config = {
        "debug_mode": True,
        "log_level": "DEBUG",
        "loaded_modules": list(logger.manager.loggerDict.keys()),
        "timestamp": datetime.utcnow()
    }
    logger.debug("Configuration requested")
    return config


@router.get("/raise_error")
async def raise_error() -> None:
    """
    Endpoint to simulate an exception for testing error handling.
    """
    logger.debug("Simulated error triggered")
    raise RuntimeError("This is a simulated error for testing purposes")
