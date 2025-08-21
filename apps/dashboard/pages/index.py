# apps/dashboard/pages/index.py
from analytics.gui_dashboard.dashboard import Dashboard

class DashboardPage:
    def __init__(self):
        self.dashboard = Dashboard()

    def render(self):
        grid_data = self.dashboard.render()
        print("=== Sentenial-X Dashboard ===")
        for section, content in grid_data.items():
            print(f"{section.upper()}: {content}")
        return grid_data
