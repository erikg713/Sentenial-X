# plugins/base_plugin.py
class Plugin:
    def run(self, data):
        raise NotImplementedError("Plugins must implement run()")

# Load plugins dynamically
import importlib.util
import os

def load_plugins():
    plugins = []
    for file in os.listdir("plugins"):
        if file.endswith(".py"):
            spec = importlib.util.spec_from_file_location("PluginModule", f"plugins/{file}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            plugins.append(mod.Plugin())
    return plugins
