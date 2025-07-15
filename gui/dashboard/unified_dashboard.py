import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QTabWidget, QLabel, QTableWidget, QTableWidgetItem
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

# Sample threat data for demonstration
SAMPLE_THREATS = [
    {"id": "T-1001", "severity": "High", "name": "Ransomware", "description": "Encrypts files", "tags": ["ransomware", "encryption"]},
    {"id": "T-1002", "severity": "Critical", "name": "Zero-Day", "description": "Unknown exploit", "tags": ["zero-day", "exploit"]},
    {"id": "T-1003", "severity": "Medium", "name": "Phishing", "description": "Credential theft", "tags": ["phishing", "social engineering"]}
]

class ThreatTable(QTableWidget):
    def __init__(self):
        super().__init__(0, 5)
        self.setHorizontalHeaderLabels(["ID", "Severity", "Name", "Description", "Tags"])
        self.setColumnWidth(3, 400)
        self.load_data(SAMPLE_THREATS)

    def load_data(self, threats):
        self.setRowCount(len(threats))
        for row, threat in enumerate(threats):
            self.setItem(row, 0, QTableWidgetItem(threat.get("id", "")))
            sev = QTableWidgetItem(threat.get("severity", "Unknown"))
            sev.setBackground(QColor(self.color_for_severity(threat.get("severity"))))
            sev.setForeground(QColor("#fff"))
            sev.setTextAlignment(Qt.AlignCenter)
            self.setItem(row, 1, sev)
            self.setItem(row, 2, QTableWidgetItem(threat.get("name", "")))
            self.setItem(row, 3, QTableWidgetItem(threat.get("description", "")))
            self.setItem(row, 4, QTableWidgetItem(", ".join(threat.get("tags", []))))

    def color_for_severity(self, severity):
        return {
            "Critical": "#d32f2f",
            "High": "#ff6b6b",
            "Medium": "#ffc107",
            "Low": "#17a2b8",
        }.get(severity, "#9e9e9e")

class PlaceholderTab(QWidget):
    def __init__(self, title):
        super().__init__()
        layout = QVBoxLayout(self)
        label = QLabel(f"{title} content goes here.")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

class UnifiedDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sentenial-X Unified Dashboard")
        self.resize(1200, 800)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tabs
        self.tabs.addTab(self.create_threat_tab(), "Threats")
        self.tabs.addTab(PlaceholderTab("Telemetry"), "Telemetry")
        self.tabs.addTab(PlaceholderTab("Simulation Controls"), "Simulation Controls")
        self.tabs.addTab(PlaceholderTab("Reports"), "Reports")
        self.tabs.addTab(PlaceholderTab("Attack Graph"), "Attack Graph")

    def create_threat_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        threat_table = ThreatTable()
        layout.addWidget(threat_table)
        return widget

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = UnifiedDashboard()
    win.show()
    sys.exit(app.exec())
