# apps/dashboard/pages/index.py
from .dashboard_page import DashboardPage

class IndexPage:
    def __init__(self):
        self.page = DashboardPage()

    def render(self):
        data = self.page.render()
        print("=== Sentenial-X Dashboard ===")
        for section, content in data.items():
            print(f"{section.upper()}: {content}")
        return data
