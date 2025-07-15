from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPlainTextEdit
from PySide6.QtCore import Qt

class TelemetryViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Telemetry Viewer")
        self.resize(600, 400)

        layout = QVBoxLayout()

        title = QLabel("Real-time Telemetry Data")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 18px; margin-bottom: 10px;")
        layout.addWidget(title)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Telemetry logs and JSON data appear here...")
        layout.addWidget(self.log_view)

        self.setLayout(layout)

    def append_log(self, text: str):
        self.log_view.appendPlainText(text)
