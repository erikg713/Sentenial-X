# sentenial-x/analytics/gui_dashboard/widgets/__init__.py
"""
Sentenial-X Dashboard Widgets
-----------------------------
Provides modular, real-time widgets for the analytics dashboard.

Widgets:
- AgentStatusWidget       : real-time agent health, heartbeats, and connectivity
- ThreatSummaryWidget     : threat frequency, severity, and anomaly charts
- CountermeasureLogWidget : RetaliationBot executed actions
- TelemetryGraphWidget    : visualizations of telemetry metrics over time
"""

from .agent_status import AgentStatusWidget
from .threat_summary import ThreatSummaryWidget
from .countermeasure_log import CountermeasureLogWidget
from .telemetry_graph import TelemetryGraphWidget
