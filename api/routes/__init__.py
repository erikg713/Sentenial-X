# -*- coding: utf-8 -*-
"""
Sentenial-X API Routes
----------------------

This module aggregates all API route modules into a single
import point for the FastAPI application.
"""

from fastapi import APIRouter

# Import route modules
from api.routes import alerts, cortex, health, threat_api

# ---------------------------------------------------------------------------
# Router aggregation
# ---------------------------------------------------------------------------
api_router = APIRouter()

# Include sub-routers
api_router.include_router(alerts.router)
api_router.include_router(cortex.router)
api_router.include_router(health.router)
api_router.include_router(threat_api.router)

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
__all__ = ["api_router"]
