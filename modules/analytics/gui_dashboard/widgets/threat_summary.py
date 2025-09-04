# analytics/gui_dashboard/widgets/threat_summary.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QLabel
from PyQt5.QtCore import QTimer

class ThreatSummaryDashboard(QWidget):
    """
    GUI widget that provides a live summary of all detected threats.
    Aggregates results from Pentest, WarGame, and Telemetry modules.
    """
    def __init__(self, parent=None, update_interval=2000):
        super().__init__(parent)
        self.setWindowTitle("Threat Summary")
        self.layout = QVBoxLayout(self)

        self.title_label = QLabel("Live Threat Summary")
        self.layout.addWidget(self.title_label)

        self.summary_display = QTextEdit()
        self.summary_display.setReadOnly(True)
        self.layout.addWidget(self.summary_display)

        self.threat_log = []

        # Timer to update summary every `update_interval` ms
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_summary)
        self.timer.start(update_interval)

    def add_threat(self, threat: dict):
        """
        Add a threat record to the summary log.
        Expected format: { 'source': 'plugin_name', 'target': 'target', 'severity': 'high', 'details': '...' }
        """
        self.threat_log.append(threat)
        self.update_summary()

    def update_summary(self):
        """
        Update the QTextEdit with aggregated threat info.
        """
        if not self.threat_log:
            self.summary_display.setPlainText("No threats detected yet.")
            return

        summary_text = ""
        for idx, threat in enumerate(self.threat_log, start=1):
            summary_text += f"{idx}. Source: {threat.get('source', 'Unknown')}\n"
            summary_text += f"   Target: {threat.get('target', 'N/A')}\n"
            summary_text += f"   Severity: {threat.get('severity', 'N/A')}\n"
            summary_text += f"   Details: {threat.get('details', 'N/A')}\n"
            summary_text += "-"*50 + "\n"

        self.summary_display.setPlainText(summary_text)
