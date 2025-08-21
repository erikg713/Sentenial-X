# apps/dashboard/services/agent_sync.py
import asyncio
from apps.dashboard.api.ingest import ingest_agent_heartbeat, ingest_telemetry
from datetime import datetime
import random

class AgentSyncService:
    """
    Simulates live agent data syncing
    """
    async def sync_loop(self):
        while True:
            agent_id = f"agent-{random.randint(1,5)}"
            ingest_agent_heartbeat(agent_id, "online", datetime.utcnow().isoformat())
            telemetry = {"cpu": random.randint(0,100), "mem": random.randint(0,100)}
            ingest_telemetry(agent_id, telemetry)
            await asyncio.sleep(5)
