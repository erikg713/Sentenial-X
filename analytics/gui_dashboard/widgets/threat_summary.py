# sentenial-x/analytics/gui_dashboard/widgets/threat_summary.py
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import requests
from ...config import API_ENDPOINTS

class ThreatSummaryWidget:
    """
    Dashboard widget showing threat summaries and analytics.
    """

    def __init__(self):
        self.id_prefix = "threat-summary"

    def render(self):
        """
        Returns the Dash component for the widget.
        """
        return html.Div(
            id=f"{self.id_prefix}-container",
            children=[
                html.H4("Threat Summary", className="mb-2"),
                dcc.Graph(
                    id=f"{self.id_prefix}-graph",
                    figure=go.Figure(),
                    style={"height": "400px"}
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
        Register callback to fetch threat data and update graph.
        """

        @app.callback(
            Output(f"{self.id_prefix}-graph", "figure"),
            Input(f"{self.id_prefix}-interval", "n_intervals")
        )
        def update_graph(n_intervals):
            try:
                response = requests.get(API_ENDPOINTS["threats"])
                threats = response.json()
                # Expected format: list of dicts with threat_type, agent_id, severity, timestamp
                df = pd.DataFrame(threats)
                if df.empty:
                    return go.Figure()

                # Aggregate threats by type
                summary = df.groupby("threat_type").size().reset_index(name="count")

                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=summary["threat_type"],
                            y=summary["count"],
                            text=summary["count"],
                            textposition="auto",
                            marker=dict(color="crimson")
                        )
                    ]
                )
                fig.update_layout(
                    template="plotly_dark",
                    margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title="Threat Type",
                    yaxis_title="Count",
                    title="Threat Frequency by Type"
                )
                return fig
            except Exception as e:
                # Return empty figure on error
                return go.Figure()
