# sentenial-x/analytics/gui_dashboard/widgets/agent_status.py
import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import requests
from ...config import API_ENDPOINTS

class AgentStatusWidget:
    """
    Real-time dashboard widget showing agent health and connectivity.
    """

    def __init__(self):
        self.id_prefix = "agent-status"

    def render(self):
        """
        Returns the Dash component for the widget.
        """
        return html.Div(
            id=f"{self.id_prefix}-container",
            children=[
                html.H4("Agent Status", className="mb-2"),
                dash_table.DataTable(
                    id=f"{self.id_prefix}-table",
                    columns=[
                        {"name": "Agent ID", "id": "agent_id"},
                        {"name": "Status", "id": "status"},
                        {"name": "Last Heartbeat", "id": "last_heartbeat"},
                        {"name": "IP Address", "id": "ip"},
                        {"name": "Location", "id": "location"},
                    ],
                    data=[],
                    style_cell={'textAlign': 'center', 'padding': '5px'},
                    style_header={'backgroundColor': 'rgb(30, 30, 30)',
                                  'color': 'white', 'fontWeight': 'bold'},
                    style_data={'backgroundColor': 'rgb(50, 50, 50)',
                                'color': 'white'},
                ),
                dcc.Interval(
                    id=f"{self.id_prefix}-interval",
                    interval=5000,  # refresh every 5 seconds
                    n_intervals=0
                )
            ]
        )

    def register_callbacks(self, app: dash.Dash):
        """
        Register callback to fetch agent data from API and update table.
        """

        @app.callback(
            Output(f"{self.id_prefix}-table", "data"),
            Input(f"{self.id_prefix}-interval", "n_intervals")
        )
        def update_table(n_intervals):
            try:
                response = requests.get(API_ENDPOINTS["agents"])
                agents = response.json()
                # Expected format: list of dicts with agent_id, status, last_heartbeat, ip, location
                df = pd.DataFrame(agents)
                return df.to_dict("records")
            except Exception as e:
                # In case API fails, return empty table
                return []
