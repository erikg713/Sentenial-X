# apps/dashboard/pages/widgets/plugin_status.py
class PluginStatusWidget:
    def __init__(self):
        self.plugins = {}

    def update_status(self, plugin_name, active):
        self.plugins[plugin_name] = active

    def render(self):
        return self.plugins
