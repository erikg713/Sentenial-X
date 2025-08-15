# sentenial-x/analytics/gui_dashboard/dashboard.py
import dash
import dash_bootstrap_components as dbc
from .layout import DashboardLayout
from .config import THEME

class Dashboard:
    """
    Main GUI dashboard for Sentenial-X analytics.
    Initializes Dash app, layout, and callbacks.
    """

    def __init__(self, title="Sentenial-X Threat Dashboard"):
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY if THEME == "dark" else dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True,
            title=title
        )
        self.layout_manager = DashboardLayout(self.app)
        self.layout_manager.register_callbacks()

    def run(self, host="0.0.0.0", port=8050, debug=False):
        """
        Launch the dashboard server.
        """
        print(f"Starting Sentenial-X dashboard on {host}:{port}...")
        self.app.run_server(host=host, port=port, debug=debug)
