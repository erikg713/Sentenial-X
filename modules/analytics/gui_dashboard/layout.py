# sentenial-x/analytics/gui_dashboard/layout.py
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from .config import DASHBOARD_TITLE, THEME, REFRESH_INTERVAL_SECONDS
from .widgets.agent_status import AgentStatusWidget
from .widgets.threat_summary import ThreatSummaryWidget
from .widgets.countermeasure_log import CountermeasureLogWidget
from .widgets.telemetry_graph import TelemetryGraphWidget

class DashboardLayout:
    """
    Manages layout and arrangement of widgets in the Sentenial-X dashboard.
    """

    def __init__(self, app: dash.Dash):
        self.app = app
        self.agent_widget = AgentStatusWidget()
        self.threat_widget = ThreatSummaryWidget()
        self.countermeasure_widget = CountermeasureLogWidget()
        self.telemetry_widget = TelemetryGraphWidget()
        self.refresh_interval = REFRESH_INTERVAL_SECONDS

        self.app.layout = self._build_layout()

    def _build_layout(self):
        """
        Construct the HTML/Bootstrap layout for the dashboard.
        """
        layout = dbc.Container(
            [
                dbc.Row(
                    dbc.Col(html.H1(DASHBOARD_TITLE, className="text-center mb-4"))
                ),
                dbc.Row(
                    [
                        dbc.Col(self.agent_widget.render(), width=3),
                        dbc.Col(self.threat_widget.render(), width=3),
                        dbc.Col(self.countermeasure_widget.render(), width=3),
                        dbc.Col(self.telemetry_widget.render(), width=3),
                    ],
                    className="mb-4"
                ),
                dcc.Interval(
                    id="interval-component",
                    interval=self.refresh_interval * 1000,  # milliseconds
                    n_intervals=0
                )
            ],
            fluid=True
        )
        return layout

    def register_callbacks(self):
        """
        Register callbacks for updating widgets on interval refresh.
        """
        self.agent_widget.register_callbacks(self.app)
        self.threat_widget.register_callbacks(self.app)
        self.countermeasure_widget.register_callbacks(self.app)
        self.telemetry_widget.register_callbacks(self.app)
