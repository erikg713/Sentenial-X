# -*- coding: utf-8 -*-
"""
Sentenial-X Main Entry Point
----------------------------

Bootstraps the FastAPI application with all API routes and optional
background services such as simulators, telemetry, and orchestrator tasks.
"""

from __future__ import annotations

import logging
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from api import create_app
from core.simulator import EmulationManager, TelemetryCollector

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("SentenialX.Main")

# ---------------------------------------------------------------------------
# Application Factory
# ---------------------------------------------------------------------------
app = create_app(title="Sentenial-X API")

# Enable CORS for local development and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Background Services
# ---------------------------------------------------------------------------
emulation_manager = EmulationManager()
telemetry_collector = TelemetryCollector()


@app.on_event("startup")
async def startup_event():
    logger.info("Starting Sentenial-X services...")
    # Start all registered simulators (non-blocking)
    emulation_manager.start_all()
    telemetry_collector.report(last_n=0)  # initialize telemetry
    logger.info("Sentenial-X startup complete.")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Sentenial-X services...")
    emulation_manager.stop_all()
    logger.info("Sentenial-X shutdown complete.")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Launching Sentenial-X API server on http://0.0.0.0:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
