# apps/dashboard/services/compliance_service.py
from apps.dashboard.pages.widgets.compliance_report import ComplianceReportWidget
import random

class ComplianceService:
    def __init__(self):
        self.widget = ComplianceReportWidget()

    async def generate_reports(self):
        agents = [f"agent-{i}" for i in range(1,6)]
        for agent in agents:
            score = random.randint(70, 100)
            self.widget.add_report(agent, "PCI-DSS", score)
            self.widget.add_report(agent, "ISO27001", score)
            self.widget.add_report(agent, "NIST", score)
        await asyncio.sleep(15)
