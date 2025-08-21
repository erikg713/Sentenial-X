# apps/dashboard/api/ingest.py
"""
Agent -> Dashboard ingestion endpoint
"""
from analytics.gui_dashboard.widgets.agent_status import AgentStatusWidget
from analytics.gui_dashboard.widgets.telemetry_graph import TelemetryGraphWidget
from analytics.gui_dashboard.widgets.threat_summary import ThreatSummaryWidget
from analytics.gui_dashboard.widgets.countermeasure_log import CountermeasureLogWidget

# Use singleton pattern for widgets
agent_status_widget = AgentStatusWidget()
telemetry_widget = TelemetryGraphWidget()
threat_widget = ThreatSummaryWidget()
countermeasure_widget = CountermeasureLogWidget()

def ingest_agent_heartbeat(agent_id: str, status: str, timestamp: str):
    agent_status_widget.update(agent_id, status, timestamp)

def ingest_telemetry(agent_id: str, telemetry: dict):
    telemetry_widget.add_data({"agent_id": agent_id, **telemetry})

def ingest_threat(agent_id: str, threat_level: str, description: str):
    threat_widget.add_threat(agent_id, threat_level, description)

def ingest_countermeasure(agent_id: str, action: str, result: str):
    countermeasure_widget.add_log(agent_id, action, result)
