from PySide6.QtWidgets import QTabWidget, QWidget, QVBoxLayout

from gui.dashboard.widgets.exploit_module import ExploitModuleWidget

class SentenialXDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sentenial-X Unified Dashboard")
        self.resize(1200, 800)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # ... existing tabs like Threats, Telemetry, etc.

        # Exploit Modules Tabs
        exploits = [
            ("ms17_010_eternalblue", "MS17-010 EternalBlue"),
            ("struts_rce", "Apache Struts RCE"),
            ("exploit_template", "Generic Exploit Template"),
        ]

        for module_name, display_name in exploits:
            tab = QWidget()
            layout = QVBoxLayout()
            exploit_widget = ExploitModuleWidget(module_name, display_name)
            layout.addWidget(exploit_widget)
            tab.setLayout(layout)
            self.tabs.addTab(tab, display_name)
