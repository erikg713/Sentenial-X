# analytics/gui_dashboard/widgets/telemetry_graph.py
class TelemetryGraphWidget:
    def __init__(self):
        self.data = []

    def add_data(self, telemetry: dict):
        self.data.append(telemetry)

    def render(self):
        # Simplified rendering, replace with charting in real frontend
        return self.data[-20:]  # Last 20 telemetry points        )

    def register_callbacks(self, app: dash.Dash):
        """
        Register callback to fetch telemetry data and update graph.
        """

        @app.callback(
            Output(f"{self.id_prefix}-graph", "figure"),
            Input(f"{self.id_prefix}-interval", "n_intervals")
        )
        def update_graph(n_intervals):
            try:
                response = requests.get(API_ENDPOINTS["telemetry"])
                data = response.json()
                # Expected format: list of dicts with agent_id, timestamp, metric_name, value
                df = pd.DataFrame(data)
                fig = go.Figure()
                if not df.empty:
                    for agent_id in df['agent_id'].unique():
                        agent_data = df[df['agent_id'] == agent_id]
                        fig.add_trace(go.Scatter(
                            x=pd.to_datetime(agent_data['timestamp']),
                            y=agent_data['value'],
                            mode='lines+markers',
                            name=f"{agent_id} - {agent_data['metric_name'].iloc[0]}"
                        ))
                fig.update_layout(
                    template="plotly_dark",
                    margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title="Time",
                    yaxis_title="Metric Value",
                    legend_title="Agents/Metrics"
                )
                return fig
            except Exception as e:
                # Return empty figure on error
                return go.Figure()
