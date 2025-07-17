import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QLabel, QComboBox, QSpinBox, QCheckBox, QTextEdit, QMessageBox
)
from PySide6.QtCore import Qt

from ransomware_emulator.emulator import RansomwareEmulator


class EmulatorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sentenial X – Ransomware Emulator")
        self.setGeometry(300, 200, 600, 400)

        self.emulator = RansomwareEmulator()

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        layout = QVBoxLayout()

        # Payload selection
        self.payload_label = QLabel("Select Payload:")
        self.payload_dropdown = QComboBox()
        self.payload_dropdown.addItems(list(self.emulator.list_payloads().keys()))

        # File count
        self.file_count_label = QLabel("Number of Test Files:")
        self.file_count_spinner = QSpinBox()
        self.file_count_spinner.setRange(1, 100)
        self.file_count_spinner.setValue(10)

        # Monitoring checkbox
        self.monitor_checkbox = QCheckBox("Enable Monitoring")
        self.monitor_checkbox.setChecked(True)

        # Run button
        self.run_button = QPushButton("Run Emulation")
        self.run_button.clicked.connect(self.run_emulation)

        # Output area
        self.output_console = QTextEdit()
        self.output_console.setReadOnly(True)
        self.output_console.setPlaceholderText("Output will appear here...")

        # Add widgets to layout
        layout.addWidget(self.payload_label)
        layout.addWidget(self.payload_dropdown)
        layout.addWidget(self.file_count_label)
        layout.addWidget(self.file_count_spinner)
        layout.addWidget(self.monitor_checkbox)
        layout.addWidget(self.run_button)
        layout.addWidget(QLabel("Results:"))
        layout.addWidget(self.output_console)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def run_emulation(self):
        payload = self.payload_dropdown.currentText()
        file_count = self.file_count_spinner.value()
        monitor = self.monitor_checkbox.isChecked()

        try:
            result = self.emulator.run_campaign(payload_name=payload, monitor=monitor, file_count=file_count)
            self.display_result(result)
        except Exception as e:
            QMessageBox.critical(self, "Execution Failed", str(e))

    def display_result(self, result: dict):
        output = "\n".join(f"{k}: {v}" for k, v in result.items())
        self.output_console.append(f"\n--- Emulation Run ---\n{output}\n")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmulatorGUI()
    window.show()
    sys.exit(app.exec())
