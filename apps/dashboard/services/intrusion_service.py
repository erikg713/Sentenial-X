# apps/dashboard/services/intrusion_service.py
from apps.dashboard.pages.widgets.intrusion_events import IntrusionEventsWidget
import random
from datetime import datetime

class IntrusionService:
    def __init__(self):
        self.widget = IntrusionEventsWidget()

    async def monitor_intrusions(self):
        agents = [f"agent-{i}" for i in range(1,6)]
        events = ["port_scan", "malware_detected", "ransomware_blocked"]
        for agent in agents:
            self.widget.add_event(agent, random.choice(events), datetime.utcnow().isoformat())
        await asyncio.sleep(6)
