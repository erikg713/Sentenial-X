import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget, QLabel
)
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt
import requests
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem


class ThreatTable(QTableWidget):
    def __init__(self):
        super().__init__(0, 5)
        self.setHorizontalHeaderLabels(["ID", "Severity", "Name", "Description", "Tags"])
        self.setColumnWidth(3, 400)

    def load_data(self, threats):
        self.setRowCount(len(threats))
        for row, threat in enumerate(threats):
            self.setItem(row, 0, QTableWidgetItem(threat.get("id", "")))

            severity = threat.get("severity", "Unknown")
            sev_item = QTableWidgetItem(severity)
            sev_item.setBackground(QColor(self.color_for_severity(severity)))
            sev_item.setForeground(QColor("#fff"))
            sev_item.setTextAlignment(Qt.AlignCenter)
            self.setItem(row, 1, sev_item)

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


class MultiModuleDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sentenial-X Unified Dashboard")
        self.resize(1400, 900)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Threat Dashboard Tab
        self.threat_tab = QWidget()
        threat_layout = QVBoxLayout(self.threat_tab)
        self.threat_table = ThreatTable()
        threat_layout.addWidget(self.threat_table)
        self.tabs.addTab(self.threat_tab, "Threat Dashboard")

        # Load threat data initially
        self.load_threats()

        # Placeholder tabs for other modules
        self.add_placeholder_tab("Ransomware Emulation")
        self.add_placeholder_tab("Pentest Suite")
        self.add_placeholder_tab("Zero-Day AI")
        self.add_placeholder_tab("Telemetry")
        self.add_placeholder_tab("Attack Graph")

    def add_placeholder_tab(self, title):
        placeholder = QWidget()
        layout = QVBoxLayout(placeholder)
        label = QLabel(f"{title} module UI coming soon...")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        self.tabs.addTab(placeholder, title)

    def load_threats(self):
        api_url = "http://localhost:5001/threats"
        try:
            import requests
            res = requests.get(api_url)
            res.raise_for_status()
            data = res.json()
            if isinstance(data, list):
                self.threat_table.load_data(data)
            else:
                print("[ERROR] Threat API did not return a list")
        except Exception as e:
            print(f"[ERROR] Failed to load threats: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MultiModuleDashboard()
    window.show()
    sys.exit(app.exec())
