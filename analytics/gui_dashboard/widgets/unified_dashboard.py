# analytics/gui_dashboard/widgets/unified_dashboard.py

from PyQt5.QtWidgets import QWidget, QTabWidget, QVBoxLayout
from .pentest_dashboard import PentestDashboard
from .war_game_dashboard import WarGameDashboard
from .agent_status import AgentStatusDashboard
from .telemetry_graph import TelemetryGraphDashboard

class UnifiedDashboard(QWidget):
    """
    Unified Sentenial-X Dashboard combining:
    - Pentest Suite
    - WarGame Plugin
    - Agent Status
    - Telemetry Graphs
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sentenial-X Unified Dashboard")
        self.layout = QVBoxLayout(self)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(PentestDashboard(), "Pentest Suite")
        self.tabs.addTab(WarGameDashboard(), "WarGame Plugin")
        self.tabs.addTab(AgentStatusDashboard(), "Agent Status")
        self.tabs.addTab(TelemetryGraphDashboard(), "Telemetry")

        self.layout.addWidget(self.tabs)
