# analytics/gui_dashboard/widgets/threat_summary.py
class ThreatSummaryWidget:
    def __init__(self):
        self.threats = []

    def add_threat(self, agent_id: str, threat_level: str, description: str):
        self.threats.append({"agent_id": agent_id, "threat_level": threat_level, "description": description})

    def render(self):
        summary = {"high": 0, "medium": 0, "low": 0}
        for t in self.threats:
            summary[t["threat_level"]] += 1
        return summary        )

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
