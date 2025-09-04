# analytics/gui_dashboard/widgets/agent_status.py

import sys
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt5.QtCore import QTimer
from core.cortex.orchestrator import AgentOrchestrator  # Assuming you have this module
from typing import List, Dict


class AgentStatusWidget(QWidget):
    """
    GUI widget to display the status of all active agents.
    Updates automatically every `update_interval_ms`.
    """

    def __init__(self, orchestrator: AgentOrchestrator, update_interval_ms: int = 2000, parent=None):
        super().__init__(parent)
        self.orchestrator = orchestrator
        self.update_interval_ms = update_interval_ms

        # Layout
        self.layout = QVBoxLayout()
        self.status_label = QLabel("Agent Status")
        self.layout.addWidget(self.status_label)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Agent ID", "Status", "Last Seen", "Current Task"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.layout.addWidget(self.table)

        self.setLayout(self.layout)

        # Timer for real-time updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_agent_status)
        self.timer.start(self.update_interval_ms)

    def update_agent_status(self):
        """
        Pull agent data from the orchestrator and refresh the table.
        """
        agents = self.fetch_agent_data()
        self.table.setRowCount(len(agents))

        for row, agent in enumerate(agents):
            self.table.setItem(row, 0, QTableWidgetItem(agent.get("agent_id", "N/A")))
            self.table.setItem(row, 1, QTableWidgetItem(agent.get("status", "Unknown")))
            self.table.setItem(row, 2, QTableWidgetItem(agent.get("last_seen", "Never")))
            self.table.setItem(row, 3, QTableWidgetItem(agent.get("current_task", "-")))

    def fetch_agent_data(self) -> List[Dict]:
        """
        Retrieve agent status from the orchestrator.
        Returns a list of dictionaries with agent info.
        """
        agent_list = []
        for agent_id, agent_obj in self.orchestrator.list_agents().items():
            agent_info = {
                "agent_id": agent_id,
                "status": agent_obj.status,
                "last_seen": agent_obj.last_seen.strftime("%Y-%m-%d %H:%M:%S") if agent_obj.last_seen else "Never",
                "current_task": agent_obj.current_task or "-",
            }
            agent_list.append(agent_info)
        return agent_list


# Example usage
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    from core.cortex.orchestrator import AgentOrchestrator

    app = QApplication(sys.argv)
    orchestrator = AgentOrchestrator()
    widget = AgentStatusWidget(orchestrator)
    widget.show()
    sys.exit(app.exec_())
