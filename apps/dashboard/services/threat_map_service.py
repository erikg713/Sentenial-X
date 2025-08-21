# apps/dashboard/services/threat_map_service.py
from apps.dashboard.pages.widgets.threat_map import ThreatMapWidget
import random

class ThreatMapService:
    def __init__(self):
        self.widget = ThreatMapWidget()

    async def update_map(self):
        agents = [f"agent-{i}" for i in range(1,6)]
        levels = ["low","medium","high"]
        for agent in agents:
            ip = f"192.168.1.{random.randint(1,254)}"
            self.widget.add_location(agent, ip, random.choice(levels))
        await asyncio.sleep(10)
