# sentenial-x/analytics/gui_dashboard/__init__.py
"""
Sentenial-X Analytics GUI Dashboard
----------------------------------
Provides a real-time interface for monitoring:
- Agent status (heartbeats, health)
- Threats (malware, XSS, SQLi)
- Countermeasures executed by RetaliationBot
- Telemetry and analytics visualizations
"""

from .dashboard import Dashboard
from .layout import DashboardLayout
from .widgets.agent_status import AgentStatusWidget
from .widgets.threat_summary import ThreatSummaryWidget
from .widgets.countermeasure_log import CountermeasureLogWidget
from .widgets.telemetry_graph import TelemetryGraphWidget
