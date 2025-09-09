"""
Sentenial-X AI Plugin System
----------------------------

Allows dynamic loading of external plugins (threat detectors, analyzers, tools).
"""
from .loader import PluginLoader

__all__ = ["PluginLoader"]
