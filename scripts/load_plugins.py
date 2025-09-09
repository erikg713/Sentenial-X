# -*- coding: utf-8 -*-
"""
scripts.load_plugins
-------------------

Dynamically discovers and loads plugin modules for Sentenial-X.

Features:
- Recursively loads all Python modules in the plugins directory.
- Registers plugin instances if they expose a `register()` function.
- Provides robust logging and error handling.
- Supports hot reloading for development purposes.
"""

from __future__ import annotations
import importlib
import logging
import os
import sys
from pathlib import Path
from typing import List, Any, Callable

# Configure module logger
logger = logging.getLogger("SentenialX.Plugins")
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

PLUGINS_DIR = Path(__file__).parent.parent / "plugins"

_loaded_plugins: List[Any] = []

def discover_plugins() -> List[Path]:
    """Discover all Python plugin files in the plugins directory."""
    if not PLUGINS_DIR.exists():
        logger.warning("Plugins directory not found: %s", PLUGINS_DIR)
        return []
    
    plugin_files = list(PLUGINS_DIR.rglob("*.py"))
    plugin_files = [p for p in plugin_files if p.name != "__init__.py"]
    logger.info("Discovered %d plugin(s).", len(plugin_files))
    return plugin_files

def load_plugin(path: Path) -> Any:
    """Load a single plugin given its file path."""
    relative_path = path.relative_to(PLUGINS_DIR.parent).with_suffix("")
    module_name = ".".join(relative_path.parts)
    try:
        module = importlib.import_module(module_name)
        # If the plugin has a register function, call it
        register_fn: Callable[..., Any] = getattr(module, "register", None)
        if callable(register_fn):
            plugin_instance = register_fn()
            _loaded_plugins.append(plugin_instance)
            logger.info("Registered plugin: %s", module_name)
        else:
            _loaded_plugins.append(module)
            logger.info("Loaded plugin module: %s", module_name)
        return module
    except Exception as e:
        logger.error("Failed to load plugin %s: %s", module_name, e)
        return None

def load_all_plugins() -> List[Any]:
    """Discover and load all plugins."""
    plugins = discover_plugins()
    loaded = []
    for plugin_path in plugins:
        module = load_plugin(plugin_path)
        if module:
            loaded.append(module)
    logger.info("Total loaded plugins: %d", len(loaded))
    return loaded

def get_loaded_plugins() -> List[Any]:
    """Return the list of already loaded plugin instances/modules."""
    return _loaded_plugins

if __name__ == "__main__":
    load_all_plugins()
