"""
api/routes/__init__.py
----------------------
Route initialization for the Sentenial-X API.
Registers all available route blueprints.
"""

from flask import Blueprint
from fastapi import FastAPI
from . import health, wormgpt, cortex, telemetry, orchestrator, alerts, ws
from . import telemetry, orchestrator, cortex, wormgpt, exploits
from api.controllers import traces_controller

def include_routes(app: FastAPI):
    app.include_router(health.router)
    app.include_router(wormgpt.router)
    app.include_router(cortex.router)
    app.include_router(telemetry.router)
    app.include_router(orchestrator.router)
    app.include_router(alerts.router)
    app.include_router(ws.router)
# Import route modules
from api.routes.monitor import monitor_bp
from api.routes.orchestrator import orchestrator_bp

# Create the main API blueprint
api_bp = Blueprint("api", __name__)

# Register sub-blueprints
api_bp.register_blueprint(monitor_bp, url_prefix="/monitor")
api_bp.register_blueprint(orchestrator_bp, url_prefix="/orchestrator")

__all__ = ["api_bp"] 
