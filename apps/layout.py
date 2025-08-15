# sentenial-x/apps/layout.py
from dash import html, dcc
import dash_bootstrap_components as dbc

class AppLayout:
    """
    Centralized layout manager for Sentenial-X apps.
    Provides a consistent header, sidebar, and content container.
    """

    def __init__(self, title="Sentenial-X"):
        self.title = title

    def render(self, content):
        """
        Wraps the provided content in a consistent layout.
        Args:
            content: Dash components to render in the main area
        Returns:
            Dash HTML layout
        """
        layout = dbc.Container(
            [
                # Header
                dbc.Row(
                    dbc.Col(
                        html.H1(self.title, className="text-center mb-4"),
                    )
                ),
                # Sidebar + Main Content
                dbc.Row(
                    [
                        # Sidebar navigation
                        dbc.Col(
                            dbc.Nav(
                                [
                                    dbc.NavLink("Dashboard", href="/dashboard", active="exact"),
                                    dbc.NavLink("Pentest Suite", href="/pentest", active="exact"),
                                    dbc.NavLink("Ransomware Emulator", href="/ransomware", active="exact"),
                                ],
                                vertical=True,
                                pills=True,
                                className="bg-dark text-light p-2",
                            ),
                            width=2,
                        ),
                        # Main content
                        dbc.Col(
                            content,
                            width=10
                        )
                    ],
                    className="mb-4"
                ),
                # Footer
                dbc.Row(
                    dbc.Col(
                        html.Footer(
                            f"© 2025 Sentenial-X - Ultimate Cyber Guardian",
                            className="text-center mt-4 mb-2 text-muted"
                        )
                    )
                )
            ],
            fluid=True
        )
        return layout
