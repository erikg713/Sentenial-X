"""
Sentenial-X Plugins Package
---------------------------

This package contains modular plugins to extend the functionality of the
Sentenial-X cybersecurity platform. Plugins can integrate with telemetry,
exploit simulation, WormGPT emulation, and other AI-driven systems.

Structure:
- base_plugin.py      : BasePlugin class for all plugins
- telemetry_plugin.py : Plugins for telemetry collection/analysis
- exploit_plugin.py   : Custom exploit simulation plugins
- wormgpt_plugin.py   : Plugins for WormGPT emulation/extensions
"""

# Expose core plugin base for external imports
from .base_plugin import BasePlugin

# Optional: preload commonly used plugins
from .telemetry_plugin import TelemetryPlugin
from .exploit_plugin import ExploitPlugin
from .wormgpt_plugin import WormGPTPlugin

__all__ = [
    "BasePlugin",
    "TelemetryPlugin",
    "ExploitPlugin",
    "WormGPTPlugin",
]
