"""
Base plugin interface.
All plugins should inherit from `BasePlugin`.
"""

class BasePlugin:
    name: str = "BasePlugin"
    description: str = "Abstract plugin base class"

    def __init__(self):
        self.enabled = True

    def run(self, *args, **kwargs):
        """Main plugin execution logic"""
        raise NotImplementedError("Plugins must implement `run()`")

    def shutdown(self):
        """Cleanup logic if needed"""
        pass
