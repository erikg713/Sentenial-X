from PySide6.QtWidgets import QTableWidget, QTableWidgetItem
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt

class ThreatTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(5)
        self.setHorizontalHeaderLabels(["ID", "Severity", "Name", "Description", "Tags"])
        self.setColumnWidth(3, 400)
        self.setSortingEnabled(True)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)

    def load_data(self, threats):
        self.setRowCount(len(threats))
        for row, threat in enumerate(threats):
            self.setItem(row, 0, QTableWidgetItem(str(threat.get("id", ""))))

            severity = threat.get("severity", "Unknown")
            sev_item = QTableWidgetItem(severity)
            sev_item.setBackground(QColor(self.color_for_severity(severity)))
            sev_item.setForeground(QColor("#FFFFFF"))
            sev_item.setTextAlignment(Qt.AlignCenter)
            self.setItem(row, 1, sev_item)

            self.setItem(row, 2, QTableWidgetItem(threat.get("name", "")))
            self.setItem(row, 3, QTableWidgetItem(threat.get("description", "")))

            tags = threat.get("tags", [])
            tags_text = ", ".join(tags) if isinstance(tags, list) else str(tags)
            self.setItem(row, 4, QTableWidgetItem(tags_text))

    def color_for_severity(self, severity):
        return {
            "Critical": "#d32f2f",
            "High": "#ff6b6b",
            "Medium": "#ffc107",
            "Low": "#17a2b8",
        }.get(severity, "#9e9e9e")
