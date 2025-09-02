"""
apps/__init__.py
----------------
Package initializer for Sentenial-X apps.

Purpose:
- Makes 'apps' a Python package.
- Optionally auto-loads application modules for CLI or API integration.
- Ensures consistent logging and configuration access across apps.
"""

# Import default configuration or logger if needed
from api.config import settings
from api.utils.logger import init_logger

logger = init_logger("apps")

# Optional: Automatically import app modules for registration
# from . import monitoring
# from . import dashboard
# from . import alerts

logger.info("Sentenial-X apps package initialized")

__all__ = [
    # "monitoring",
    # "dashboard",
    # "alerts",
]
