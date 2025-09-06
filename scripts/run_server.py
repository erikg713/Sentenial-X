# -*- coding: utf-8 -*-
"""
Run Sentenial-X API Server
--------------------------

- Starts FastAPI backend
- Supports development and production modes
- Includes logging and graceful shutdown
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI

from core.simulator import EmulationManager, TelemetryCollector
from core.simulator.wormgpt_clone import WormGPTDetector
from core.simulator.blind_spot_tracker import BlindSpotTracker

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SentenialX.Server")

# ---------------------------------------------------------------------------
# FastAPI App Initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Sentenial-X API",
    description="Sentenial-X Threat Simulation and Analysis API",
    version="1.0.0"
)

# Initialize core managers
emulation_manager = EmulationManager()
telemetry_collector = TelemetryCollector()

# Register default simulators
emulation_manager.register(WormGPTDetector())
emulation_manager.register(BlindSpotTracker())

# ---------------------------------------------------------------------------
# Basic API Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "Sentenial-X API Server is running!"}

@app.get("/simulators")
async def list_simulators():
    sims = [sim.name for sim in emulation_manager.simulators]
    return {"simulators": sims}

@app.post("/simulate/{sim_name}")
async def run_simulator(sim_name: str):
    simulator = next((s for s in emulation_manager.simulators if s.name == sim_name), None)
    if not simulator:
        return {"error": f"Simulator '{sim_name}' not found"}
    result = simulator.run()
    return {"simulator": sim_name, "result": result}

@app.get("/telemetry")
async def get_telemetry(last_n: Optional[int] = 10):
    data = telemetry_collector.report(last_n=last_n)
    return {"telemetry": data}

# ---------------------------------------------------------------------------
# Server runner
# ---------------------------------------------------------------------------
def main(host: str = "127.0.0.1", port: int = 8000, reload: bool = True):
    """
    Start the FastAPI server using uvicorn
    """
    logger.info("Starting Sentenial-X API Server on %s:%s", host, port)

    # Graceful shutdown handler
    def shutdown_handler(signum, frame):
        logger.info("Shutting down Sentenial-X server...")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    uvicorn.run(
        "scripts.run_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Sentenial-X API server.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default 8000)")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload (production)")
    args = parser.parse_args()

    main(host=args.host, port=args.port, reload=not args.no_reload)
