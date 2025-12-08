"""
Sentenial X AI - Plugins Management

This package is designed to manage dynamic loading, registration, and
unloading of external modules (plugins) to extend the agent's capabilities
without modifying the core system.
"""

import logging
from typing import Dict, Any, Callable, Optional, List

logger = logging.getLogger(__name__)

# Registry to store information about loaded plugins
# Format: { plugin_name: { 'module': module_object, 'description': str, ... } }
_PLUGIN_REGISTRY: Dict[str, Dict[str, Any]] = {}

class PluginError(Exception):
    """Base exception for plugin-related errors."""
    pass

def register_plugin(name: str, description: str = "", requirements: Optional[List[str]] = None):
    """
    Decorator to register a class or function as a plugin entry point.
    
    In a fully featured system, this would store meta-data about the plugin.
    For now, it acts as a simple marker and registry hook.

    Args:
        name: The unique name of the plugin (e.g., 'data_analysis_tool').
        description: A brief description of the plugin's function.
        requirements: A list of dependencies needed by the plugin.
    """
    def decorator(plugin_entry: Callable):
        plugin_info = {
            "entry_point": plugin_entry,
            "description": description,
            "name": name,
            "requirements": requirements or [],
            "status": "registered"
        }
        
        if name in _PLUGIN_REGISTRY:
            logger.warning(f"Overwriting existing plugin registration for '{name}'")
        
        _PLUGIN_REGISTRY[name] = plugin_info
        logger.info(f"Plugin '{name}' registered successfully.")
        return plugin_entry
    return decorator

def get_loaded_plugins() -> Dict[str, str]:
    """Returns a dictionary of loaded plugin names and their descriptions."""
    return {name: info['description'] for name, info in _PLUGIN_REGISTRY.items()}

# In a complete system, methods for dynamic loading (importing modules 
# from a 'plugins/' directory) and execution would be implemented here.

__all__ = [
    "register_plugin",
    "get_loaded_plugins",
    "PluginError"
]
