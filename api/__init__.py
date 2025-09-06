# -*- coding: utf-8 -*-
"""
API Package Initializer for Sentenial-X
---------------------------------------

Provides central access to all API routes and optional global resources.
"""

from __future__ import annotations

from fastapi import FastAPI
from api.routes import router as api_router

__all__ = ["create_app"]


def create_app(title: str = "Sentenial-X API") -> FastAPI:
    """
    Factory function to create and configure a FastAPI application
    with all routes registered.
    """
    app = FastAPI(title=title, version="1.0.0")

    # Include all registered routes
    app.include_router(api_router)

    # Optional: Add middleware, exception handlers, or startup events here
    @app.on_event("startup")
    async def startup_event():
        # Example: initialize telemetry collector or logging
        import logging
        logging.getLogger("SentenialX").info("Sentenial-X API startup complete")

    @app.on_event("shutdown")
    async def shutdown_event():
        logging.getLogger("SentenialX").info("Sentenial-X API shutdown complete")

    return app
