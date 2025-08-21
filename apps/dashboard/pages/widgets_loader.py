# apps/dashboard/pages/widgets_loader.py
from apps.dashboard.pages.dashboard_page import DashboardPage

class WidgetsLoader:
    def __init__(self):
        self.dashboard = DashboardPage()

    def load_all(self):
        return self.dashboard.render()
