# apps/dashboard/pages/live_dashboard.py
import asyncio
from apps.dashboard.pages.Home import HomePage

class LiveDashboard:
    def __init__(self):
        self.home = HomePage()

    async def run(self, interval=5):
        while True:
            self.home.render()
            await asyncio.sleep(interval)
