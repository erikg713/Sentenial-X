"""
api/routes/__init__.py
----------------------
Route initialization for the Sentenial-X API.
Registers all available route blueprints.
"""

from flask import Blueprint

# Import route modules
from api.routes.monitor import monitor_bp
from api.routes.orchestrator import orchestrator_bp

# Create the main API blueprint
api_bp = Blueprint("api", __name__)

# Register sub-blueprints
api_bp.register_blueprint(monitor_bp, url_prefix="/monitor")
api_bp.register_blueprint(orchestrator_bp, url_prefix="/orchestrator")

__all__ = ["api_bp"] 