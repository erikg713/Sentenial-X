# apps/dashboard/layout.py
from analytics.gui_dashboard.layout import Layout

class DashboardLayout:
    def __init__(self):
        self.layout = Layout()

    def get_grid(self):
        return self.layout.render()
