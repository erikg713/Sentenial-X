# apps/dashboard/main.py
import asyncio
from apps.dashboard.services.agent_sync import AgentSyncService
from apps.dashboard.services.alert_service import AlertService
from apps.dashboard.services.threat_service import ThreatService
from apps.dashboard.pages.index import DashboardPage

async def run_dashboard():
    # Initialize dashboard page
    page = DashboardPage()
    # Initialize services
    agent_sync = AgentSyncService()
    alert_service = AlertService()
    threat_service = ThreatService()
    
    # Run loops concurrently
    await asyncio.gather(
        agent_sync.sync_loop(),
        alert_service.alert_loop(),
        threat_service.threat_loop()
    )

if __name__ == "__main__":
    asyncio.run(run_dashboard())
