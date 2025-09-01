#!/usr/bin/env python3
"""
cli/__init__.py

Sentenial-X CLI package initializer.
Allows importing CLI modules like wormgpt, telemetry, alerts, orchestrator.
"""

__version__ = "1.0.0"
__author__ = "Sentenial-X Dev Team"

# Optional: import core CLI handlers for convenience
from .wormgpt import run_wormgpt, handle_wormgpt
# from .telemetry import run_telemetry, handle_telemetry
# from .alerts import dispatch_alert, handle_alert
# from .orchestrator import run_orchestrator, handle_orchestrator

# You can also define package-level helpers here
import asyncio

def run_cli_handler(handler, args):
    """
    Utility to run async CLI handlers in a synchronous entrypoint.
    Example:
        run_cli_handler(handle_wormgpt, args)
    """
    return asyncio.run(handler(args))
