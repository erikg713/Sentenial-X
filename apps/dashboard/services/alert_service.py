# apps/dashboard/services/alert_service.py
import random
from apps.dashboard.api.ingest import ingest_countermeasure

class AlertService:
    """
    Simulated alert dispatcher from agents
    """
    async def alert_loop(self):
        while True:
            agent_id = f"agent-{random.randint(1,5)}"
            action = random.choice(["blocked_ransomware", "wormgpt_detected"])
            result = f"{action} executed"
            ingest_countermeasure(agent_id, action, result)
            await asyncio.sleep(7)
