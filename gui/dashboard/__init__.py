"""
Sentenial X :: Dashboard Package

This module initializes the dashboard package and exposes commonly used components.
"""

from pathlib import Path
import logging

# Set up a package-level logger
logger = logging.getLogger("sentenialx.dashboard")
logger.setLevel(logging.INFO)

# Example: Path to static resources (optional)
DASHBOARD_ROOT = Path(__file__).resolve().parent
STATIC_PATH = DASHBOARD_ROOT / "static"

__all__ = ["logger", "STATIC_PATH"]

