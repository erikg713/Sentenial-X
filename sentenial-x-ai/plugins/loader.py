"""
Dynamic plugin loader.
Scans the `plugins/` directory and loads Python modules.
"""

import importlib
import pkgutil
import pathlib

from .base import BasePlugin

class PluginLoader:
    def __init__(self, plugin_path: str = None):
        self.plugin_path = plugin_path or str(pathlib.Path(__file__).parent)
        self.plugins = {}

    def discover(self):
        """Find all available plugins in the folder."""
        package = "sentenial-x-ai.plugins"
        for _, name, is_pkg in pkgutil.iter_modules([self.plugin_path]):
            if not is_pkg and name not in ("base", "loader", "__init__"):
                yield name

    def load_all(self):
        """Load all discovered plugins."""
        for plugin_name in self.discover():
            self.load(plugin_name)

    def load(self, plugin_name: str):
        """Load a single plugin by name."""
        try:
            module = importlib.import_module(f"sentenial-x-ai.plugins.{plugin_name}")
            plugin_class = None

            # Find subclass of BasePlugin
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, BasePlugin) and attr is not BasePlugin:
                    plugin_class = attr
                    break

            if plugin_class:
                instance = plugin_class()
                self.plugins[plugin_name] = instance
                print(f"[PLUGIN] Loaded: {plugin_class.__name__}")
                return instance
            else:
                print(f"[PLUGIN] No valid BasePlugin subclass found in {plugin_name}")
        except Exception as e:
            print(f"[PLUGIN] Failed to load {plugin_name}: {e}")

    def run_all(self, *args, **kwargs):
        """Execute all loaded plugins."""
        for name, plugin in self.plugins.items():
            if plugin.enabled:
                print(f"[PLUGIN] Running {plugin.name}...")
                plugin.run(*args, **kwargs)
