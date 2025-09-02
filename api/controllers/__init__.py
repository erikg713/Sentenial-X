"""
API Controllers Package
-----------------------

This package contains the controller modules for the Sentenial-X API.

Each controller is responsible for handling business logic tied to
specific endpoints (auth, telemetry, chain-executor, etc.).

Controllers should:
- Receive validated requests from FastAPI routes
- Interact with database models or services
- Return responses or raise appropriate exceptions
"""

from fastapi import APIRouter

# Create a global router to include all controllers
router = APIRouter()

# Import controllers here and register them with the router
# Example:
# from . import auth_controller, telemetry_controller
# router.include_router(auth_controller.router, prefix="/auth", tags=["Auth"])
# router.include_router(telemetry_controller.router, prefix="/telemetry", tags=["Telemetry"])
