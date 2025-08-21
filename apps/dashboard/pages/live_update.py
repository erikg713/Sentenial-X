# apps/dashboard/pages/live_update.py
import asyncio
from apps.dashboard.pages.widgets_loader import WidgetsLoader

class LiveUpdate:
    def __init__(self):
        self.loader = WidgetsLoader()

    async def update_loop(self, interval=5):
        while True:
            data = self.loader.load_all()
            # Here, you could push via websocket to frontend
            await asyncio.sleep(interval)
