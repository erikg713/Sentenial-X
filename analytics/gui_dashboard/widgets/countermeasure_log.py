# analytics/gui_dashboard/widgets/countermeasure_log.py
class CountermeasureLogWidget:
    def __init__(self):
        self.logs = []

    def add_log(self, agent_id: str, action: str, result: str):
        self.logs.append({"agent_id": agent_id, "action": action, "result": result})

    def render(self):
        return self.logs[-10:]  # Show last 10 logs                                  'color': 'white', 'fontWeight': 'bold'},
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
