 """
Sentenial-X API Package Initialization.

This package exposes REST and WebSocket APIs for interacting with the
Sentenial-X system. It handles external requests, routes, and provides
secure access to core functionality including orchestrator, telemetry,
and AI-driven modules.
"""

from fastapi import FastAPI

# Global API app instance
app = FastAPI(
    title="Sentenial-X API",
    description="REST and WebSocket APIs for the Sentenial-X cybersecurity suite.",
    version="1.0.0"
)

# Import routes so they get registered
try:
    from . import routes  # noqa: F401
except ImportError:
    # Routes may not exist yet during initial setup
    pass


def get_app() -> FastAPI:
    """
    Returns the FastAPI application instance for use in ASGI servers.
    """
    return app
