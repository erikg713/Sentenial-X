# apps/dashboard/api/agents_api.py
from typing import List, Dict
from analytics.gui_dashboard.widgets.agent_status import AgentStatusWidget
from analytics.gui_dashboard.widgets.telemetry_graph import TelemetryGraphWidget
from analytics.gui_dashboard.widgets.threat_summary import ThreatSummaryWidget
from analytics.gui_dashboard.widgets.countermeasure_log import CountermeasureLogWidget

# Global widget instances (simulating backend DB or in-memory store)
agent_status_widget = AgentStatusWidget()
telemetry_widget = TelemetryGraphWidget()
threat_widget = ThreatSummaryWidget()
countermeasure_widget = CountermeasureLogWidget()

def fetch_agent_status() -> Dict:
    return agent_status_widget.render()

def fetch_telemetry() -> List[dict]:
    return telemetry_widget.render()

def fetch_threats() -> dict:
    return threat_widget.render()

def fetch_countermeasures() -> list:
    return countermeasure_widget.render()
