"""
libs/core/__init__.py

Core utilities and shared modules for Sentenial-X platform.
Provides centralized imports for environment handling, configuration, and other core services.
"""

# Expose key core modules
from libs.core import config_loader
from libs.core import env_handler

# Optional: define default exports for easier imports
__all__ = [
    "config_loader",
    "env_handler",
]
