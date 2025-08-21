# apps/dashboard/live_dashboard_runner.py
import asyncio
from services.agent_summary_service import AgentSummaryService
from services.network_monitor import NetworkMonitor
from services.system_alerts_service import SystemAlertsService
from services.intrusion_service import IntrusionService
from services.security_score_service import SecurityScoreService
from services.compliance_service import ComplianceService
from services.exploit_monitor import ExploitMonitor
from services.threat_map_service import ThreatMapService
from plugin_manager import PluginManager
from pages.Home import HomePage

class DashboardOrchestrator:
    """
    Orchestrates all dashboard services for live updates.
    """
    def __init__(self):
        # Widgets / services
        self.agent_summary_service = AgentSummaryService()
        self.network_monitor = NetworkMonitor()
        self.system_alerts = SystemAlertsService()
        self.intrusion_service = IntrusionService()
        self.security_score_service = SecurityScoreService()
        self.compliance_service = ComplianceService()
        self.exploit_monitor = ExploitMonitor()
        self.threat_map_service = ThreatMapService()
        self.plugin_manager = PluginManager()

        # HomePage aggregator
        self.home = HomePage()

    async def run_services(self):
        """
        Schedule all services concurrently
        """
        tasks = [
            self.agent_summary_service.update_summary(),
            self.network_monitor.update_traffic(),
            self.system_alerts.generate_alerts(),
            self.intrusion_service.monitor_intrusions(),
            self.security_score_service.update_scores(),
            self.compliance_service.generate_reports(),
            self.exploit_monitor.simulate_exploits(),
            self.threat_map_service.update_map()
        ]
        await asyncio.gather(*tasks)

    async def run_live_dashboard(self, interval=5):
        """
        Continuously update and render dashboard every interval seconds
        """
        while True:
            await self.run_services()
            self.home.render()
            await asyncio.sleep(interval)

# CLI Entrypoint
if __name__ == "__main__":
    orchestrator = DashboardOrchestrator()
    try:
        asyncio.run(orchestrator.run_live_dashboard(interval=5))
    except KeyboardInterrupt:
        print("\n[Live Dashboard terminated by user]")
