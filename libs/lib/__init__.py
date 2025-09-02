"""
libs/lib/__init__.py

Core utilities and helper library package for Sentenial-X.
This package provides shared functionality for plugins, agents, and AI modules.
"""

# Expose key submodules for easy imports
from . import utils
from . import security
from . import io
from . import network

# Optional: package version
__version__ = "1.0.0"

# Optional: package logger
import logging

logger = logging.getLogger("sentenialx.lib")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

logger.info("libs.lib package initialized")
