import importlib
import os

PLUGINS = {}

def load_plugins(plugin_dir="plugins"):
    for fname in os.listdir(plugin_dir):
        if fname.endswith(".py") and not fname.startswith("__"):
            module_name = f"{plugin_dir.replace('/', '.')}.{fname[:-3]}"
            module = importlib.import_module(module_name)

            if hasattr(module, "register"):
                plugin = module.register()
                PLUGINS[plugin.name] = plugin

def get_plugin_parameters(plugin_name):
    return PLUGINS[plugin_name].parameters

def run_plugin(plugin_name, **kwargs):
    return PLUGINS[plugin_name].run(**kwargs)
if __name__ == "__main__":
    load_plugins()

    params = get_plugin_parameters("ransomware_emulator")
    print("Parameters:", params)

    result = run_plugin("ransomware_emulator", payload_name="test_payload", file_count=3, monitor=True)
    print("Run Result:", result)
