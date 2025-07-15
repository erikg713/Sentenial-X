import sys
import requests
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QTableWidget, QTableWidgetItem
)
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt


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


class SentenialXDashboard(QMainWindow):
    def __init__(self, api_url="http://localhost:5001/threats"):
        super().__init__()
        self.setWindowTitle("Sentenial X :: Threat Dashboard")
        self.resize(1280, 800)

        self.api_url = api_url
        self.table = ThreatTable()

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.table)
        self.setCentralWidget(central_widget)

        self.load_data()

    def load_data(self):
        try:
            res = requests.get(self.api_url)
            res.raise_for_status()
            threats = res.json()
            if isinstance(threats, list):
                self.table.load_data(threats)
            else:
                print("[ERROR] API did not return a list.")
        except Exception as e:
            print(f"[ERROR] Failed to fetch threats: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dashboard = SentenialXDashboard()
    dashboard.show()
    sys.exit(app.exec())
