# apps/dashboard/services/security_score_service.py
from apps.dashboard.pages.widgets.security_score import SecurityScoreWidget
import random

class SecurityScoreService:
    def __init__(self):
        self.widget = SecurityScoreWidget()

    async def update_scores(self):
        agents = [f"agent-{i}" for i in range(1,6)]
        for agent in agents:
            self.widget.update_score(agent, random.randint(50,100))
        await asyncio.sleep(10)
