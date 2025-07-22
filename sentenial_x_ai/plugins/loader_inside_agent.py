import importlib
import os

from executor import register_custom_command

def load_plugins():
    for fname in os.listdir("plugins"):
        if not fname.endswith(".py") or fname == "__init__.py":
            continue
        module_name = f"plugins.{fname[:-3]}"
        mod = importlib.import_module(module_name)
        if hasattr(mod, "register"):
            mod.register({
                "register_command": register_custom_command
            })
