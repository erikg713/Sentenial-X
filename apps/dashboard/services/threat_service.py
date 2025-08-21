# apps/dashboard/services/threat_service.py
import random
from apps.dashboard.api.ingest import ingest_threat

class ThreatService:
    """
    Simulated threat ingestion for dashboard
    """
    async def threat_loop(self):
        levels = ["low", "medium", "high"]
        while True:
            agent_id = f"agent-{random.randint(1,5)}"
            threat_level = random.choice(levels)
            description = "Simulated threat event"
            ingest_threat(agent_id, threat_level, description)
            await asyncio.sleep(10)
