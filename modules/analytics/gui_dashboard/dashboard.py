# analytics/gui_dashboard/dashboard.py
from .layout import Layout
from .widgets import agent_status, telemetry_graph, threat_summary, countermeasure_log

class Dashboard:
    def __init__(self):
        self.layout = Layout()
        self.widgets = {
            "agent_status": agent_status.AgentStatusWidget(),
            "telemetry_graph": telemetry_graph.TelemetryGraphWidget(),
            "threat_summary": threat_summary.ThreatSummaryWidget(),
            "countermeasure_log": countermeasure_log.CountermeasureLogWidget()
        }

    def render(self):
        grid = self.layout.grid
        rendered = {}
        for key, widget_name in grid.items():
            widget = self.widgets.get(widget_name)
            rendered[key] = widget.render() if widget else "Empty"
        return rendered
