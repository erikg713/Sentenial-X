# analytics/gui_dashboard/widgets/agent_status.py
from datetime import datetime

class AgentStatusWidget:
    def __init__(self):
        self.statuses = {}

    def update(self, agent_id: str, status: str, timestamp: str):
        self.statuses[agent_id] = {"status": status, "timestamp": timestamp}

    def render(self):
        return {agent: f"{data['status']} @ {data['timestamp']}" for agent, data in self.statuses.items()}                                  'color': 'white', 'fontWeight': 'bold'},
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
