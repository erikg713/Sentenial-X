# -*- coding: utf-8 -*-
"""
Load and register Sentenial-X plugins
--------------------------------------

- Dynamically discovers Python plugins in the `plugins/` directory
- Loads each plugin as a module
- Registers plugin engines with EmulationManager
- Supports metadata inspection (name, version, author)
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from core.simulator import EmulationManager, BaseEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SentenialX.PluginsLoader")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PLUGIN_DIR = Path(__file__).parent.parent / "plugins"

# ---------------------------------------------------------------------------
# Plugin discovery
# ---------------------------------------------------------------------------
def discover_plugins(plugin_dir: Path = PLUGIN_DIR) -> List[Path]:
    """Return a list of plugin Python files."""
    if not plugin_dir.exists():
        logger.warning("Plugin directory does not exist: %s", plugin_dir)
        return []

    plugins = [p for p in plugin_dir.glob("*.py") if not p.name.startswith("_")]
    logger.info("Discovered %d plugins in %s", len(plugins), plugin_dir)
    return plugins

# ---------------------------------------------------------------------------
# Load plugin module
# ---------------------------------------------------------------------------
def load_plugin_module(plugin_path: Path) -> Any:
    """Import a plugin module dynamically."""
    module_name = f"plugins.{plugin_path.stem}"
    if str(plugin_path.parent) not in sys.path:
        sys.path.insert(0, str(plugin_path.parent))
    try:
        module = importlib.import_module(module_name)
        logger.info("Loaded plugin module: %s", module_name)
        return module
    except Exception as exc:
        logger.exception("Failed to load plugin %s: %s", module_name, exc)
        return None

# ---------------------------------------------------------------------------
# Register plugin with EmulationManager
# ---------------------------------------------------------------------------
def register_plugins(manager: EmulationManager, plugin_dir: Path = PLUGIN_DIR) -> List[str]:
    """Load all plugins and register engines with the EmulationManager."""
    loaded_plugins = []
    for plugin_path in discover_plugins(plugin_dir):
        module = load_plugin_module(plugin_path)
        if module is None:
            continue

        # Look for `engine` or `Engine` class/factory
        engine = getattr(module, "engine", None) or getattr(module, "Engine", None) or getattr(module, "create_engine", None)
        if engine is None:
            logger.warning("No engine found in plugin: %s", plugin_path.stem)
            continue

        # Instantiate if needed
        if callable(engine) and not isinstance(engine, BaseEngine):
            try:
                engine = engine()
            except Exception as exc:
                logger.exception("Failed to instantiate engine in plugin %s: %s", plugin_path.stem, exc)
                continue

        if isinstance(engine, BaseEngine):
            manager.register(engine)
            loaded_plugins.append(plugin_path.stem)
            logger.info("Plugin engine registered: %s", plugin_path.stem)
        else:
            logger.warning("Engine in plugin %s is not a BaseEngine instance", plugin_path.stem)

    return loaded_plugins

# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def main():
    from core.simulator import EmulationManager

    manager = EmulationManager()
    plugins = register_plugins(manager)
    logger.info("Total plugins loaded and registered: %d", len(plugins))
    print("Loaded Plugins:", plugins)


if __name__ == "__main__":
    main()
