"""
apps/dashboard/__init__.py
--------------------------
Dashboard module initializer for Sentenial-X.

Purpose:
- Makes 'dashboard' a Python package.
- Initializes dashboard-related services and logging.
- Can be extended to auto-load widgets, controllers, or UI components.
"""

from api.config import settings
from api.utils.logger import init_logger

logger = init_logger("dashboard")

# Optional: import submodules or controllers for auto-registration
# from . import widgets
# from . import controllers
# from . import services

logger.info("Dashboard module initialized")

__all__ = [
    # "widgets",
    # "controllers",
    # "services",
]
