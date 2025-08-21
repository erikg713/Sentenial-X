# apps/dashboard/services/agent_summary_service.py
from apps.dashboard.pages.widgets.agent_summary import AgentSummaryWidget
import random

class AgentSummaryService:
    def __init__(self):
        self.widget = AgentSummaryWidget()

    async def update_summary(self):
        agents = [f"agent-{i}" for i in range(1,6)]
        for agent in agents:
            metrics = {
                "cpu": random.randint(0,100),
                "memory": random.randint(0,100),
                "disk": random.randint(0,100)
            }
            self.widget.update_summary(agent, metrics)
        await asyncio.sleep(5)
