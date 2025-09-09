# analytics/gui_dashboard/widgets/countermeasure_log.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QLabel
from PyQt5.QtCore import QTimer

class CountermeasureLogDashboard(QWidget):
    """
    GUI widget to display live countermeasure logs.
    Tracks automated responses to detected threats.
    """
    def __init__(self, parent=None, update_interval=2000):
        super().__init__(parent)
        self.setWindowTitle("Countermeasure Log")
        self.layout = QVBoxLayout(self)

        self.title_label = QLabel("Live Countermeasure Log")
        self.layout.addWidget(self.title_label)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.layout.addWidget(self.log_display)

        self.countermeasure_log = []

        # Timer to refresh log display
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_log_display)
        self.timer.start(update_interval)

    def add_countermeasure(self, countermeasure: dict):
        """
        Add a countermeasure entry to the log.
        Expected format: { 'action': 'blocked IP', 'source': 'plugin_name', 'target': 'target', 'details': '...' }
        """
        self.countermeasure_log.append(countermeasure)
        self.update_log_display()

    def update_log_display(self):
        """
        Refresh the QTextEdit with the latest countermeasure log.
        """
        if not self.countermeasure_log:
            self.log_display.setPlainText("No countermeasures executed yet.")
            return

        display_text = ""
        for idx, cm in enumerate(self.countermeasure_log, start=1):
            display_text += f"{idx}. Action: {cm.get('action', 'Unknown')}\n"
            display_text += f"   Source: {cm.get('source', 'N/A')}\n"
            display_text += f"   Target: {cm.get('target', 'N/A')}\n"
            display_text += f"   Details: {cm.get('details', 'N/A')}\n"
            display_text += "-"*50 + "\n"

        self.log_display.setPlainText(display_text)
