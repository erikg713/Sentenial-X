from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton

class SimulationControls(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Simulation")
        self.pause_btn = QPushButton("Pause Simulation")
        self.stop_btn = QPushButton("Stop Simulation")

        layout.addWidget(self.start_btn)
        layout.addWidget(self.pause_btn)
        layout.addWidget(self.stop_btn)

        self.setLayout(layout)
