import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget

# Import all your tool UIs
from threat_collector_gui import ThreatCollectorUI
from sentenial_ui import SentenialUI
from sentenial_service import SentenialServiceUI
from sentenial_core import SentenialCoreUI

class Hub(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Centennial X - Unified Interface")
        self.resize(1200, 800)

        tabs = QTabWidget()

        tabs.addTab(ThreatCollectorUI(), "Threat Collector")
        tabs.addTab(SentenialUI(), "Sentenial UI")
        tabs.addTab(SentenialCoreUI(), "Sentenial Core")
        tabs.addTab(SentenialServiceUI(), "Sentenial Service")

        self.setCentralWidget(tabs)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Hub()
    window.show()
    sys.exit(app.exec())