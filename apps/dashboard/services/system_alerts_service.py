# apps/dashboard/services/system_alerts_service.py
from apps.dashboard.pages.widgets.system_alerts import SystemAlertsWidget
import random

class SystemAlertsService:
    def __init__(self):
        self.widget = SystemAlertsWidget()

    async def generate_alerts(self):
        agents = [f"agent-{i}" for i in range(1,6)]
        severities = ["low","medium","high"]
        messages = ["CPU spike","Memory leak","Unauthorized access"]
        for agent in agents:
            self.widget.add_alert(agent, random.choice(messages), random.choice(severities))
        await asyncio.sleep(7)
