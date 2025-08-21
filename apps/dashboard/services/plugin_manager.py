# apps/dashboard/services/plugin_manager.py
class PluginManager:
    """
    Tracks active Pentest Suite plugins and their status
    """
    def __init__(self):
        self.active_plugins = {}

    def activate_plugin(self, name):
        self.active_plugins[name] = True

    def deactivate_plugin(self, name):
        self.active_plugins[name] = False

    def get_status(self):
        return self.active_plugins
