# sentenial-x/analytics/gui_dashboard/widgets/countermeasure_log.py
import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import requests
from ...config import API_ENDPOINTS

class CountermeasureLogWidget:
    """
    Dashboard widget displaying RetaliationBot executed actions in real-time.
    """

    def __init__(self):
        self.id_prefix = "countermeasure-log"

    def render(self):
        """
        Returns the Dash component for the widget.
        """
        return html.Div(
            id=f"{self.id_prefix}-container",
            children=[
                html.H4("Countermeasure Log", className="mb-2"),
                dash_table.DataTable(
                    id=f"{self.id_prefix}-table",
                    columns=[
                        {"name": "Timestamp", "id": "timestamp"},
                        {"name": "Agent ID", "id": "agent_id"},
                        {"name": "Threat Detected", "id": "threat"},
                        {"name": "Action Taken", "id": "action"},
                        {"name": "Status", "id": "status"},
                    ],
                    data=[],
                    style_cell={'textAlign': 'center', 'padding': '5px'},
                    style_header={'backgroundColor': 'rgb(30, 30, 30)',
                                  'color': 'white', 'fontWeight': 'bold'},
                    style_data={'backgroundColor': 'rgb(50, 50, 50)',
                                'color': 'white'},
                    style_table={'overflowY': 'auto', 'maxHeight': '400px'}
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
        Register callback to fetch countermeasure logs from API and update table.
        """

        @app.callback(
            Output(f"{self.id_prefix}-table", "data"),
            Input(f"{self.id_prefix}-interval", "n_intervals")
        )
        def update_table(n_intervals):
            try:
                response = requests.get(API_ENDPOINTS["countermeasures"])
                logs = response.json()
                # Expected format: list of dicts with timestamp, agent_id, threat, action, status
                df = pd.DataFrame(logs)
                df = df.sort_values(by="timestamp", ascending=False)  # latest first
                return df.to_dict("records")
            except Exception as e:
                # If API fails, return empty list
                return []
