# apps/dashboard/services/network_monitor.py
from apps.dashboard.pages.widgets.network_traffic import NetworkTrafficWidget
import random

class NetworkMonitor:
    def __init__(self):
        self.widget = NetworkTrafficWidget()

    async def update_traffic(self):
        agents = [f"agent-{i}" for i in range(1,6)]
        for agent in agents:
            sent = random.randint(1000,100000)
            recv = random.randint(1000,100000)
            self.widget.add_log(agent, sent, recv)
        await asyncio.sleep(8)
