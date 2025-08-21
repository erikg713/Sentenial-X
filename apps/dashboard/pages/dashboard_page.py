# apps/dashboard/pages/dashboard_page.py
from apps.dashboard.widgets.agent_card import AgentCardWidget
from apps.dashboard.widgets.telemetry_chart import TelemetryChartWidget
from apps.dashboard.widgets.threat_panel import ThreatPanelWidget
from apps.dashboard.widgets.countermeasure_log import CountermeasureLogWidget

class DashboardPage:
    def __init__(self):
        self.agent_card = AgentCardWidget()
        self.telemetry_chart = TelemetryChartWidget()
        self.threat_panel = ThreatPanelWidget()
        self.countermeasure_log = CountermeasureLogWidget()

    def render(self):
        return {
            "agents": self.agent_card.render(),
            "telemetry": self.telemetry_chart.render(),
            "threat_summary": self.threat_panel.render(),
            "countermeasure_log": self.countermeasure_log.render()
        }
