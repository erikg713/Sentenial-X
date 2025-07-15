import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout
from gui.dashboard.widgets.threat_table import ThreatTable  # Your existing widget imports
from gui.dashboard.widgets.telemetry_viewer import TelemetryViewer
from gui.dashboard.widgets.simulation_controls import SimulationControls
from gui.dashboard.widgets.report_card import ReportCard
from gui.dashboard.widgets.attack_graph import AttackGraph
from gui.dashboard.widgets.exploit_module import ExploitModuleWidget
from gui.dashboard.widgets.exploit_module import ExploitModuleTab
# ...
self.tabs.addTab(ExploitModuleTab(), "Exploits")

# Sample threats data
SAMPLE_THREATS = [
    {"id": "T-1001", "severity": "High", "name": "Ransomware", "description": "Encrypts files", "tags": ["ransomware", "encryption"]},
    {"id": "T-1002", "severity": "Critical", "name": "Zero-Day", "description": "Unknown exploit", "tags": ["zero-day", "exploit"]},
    {"id": "T-1003", "severity": "Medium", "name": "Phishing", "description": "Credential theft", "tags": ["phishing", "social engineering"]},
]

class SentenialXDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sentenial-X Unified Dashboard")
        self.resize(1200, 800)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Threats Tab
        threat_tab = QWidget()
        threat_layout = QVBoxLayout()
        self.threat_table = ThreatTable()
        self.threat_table.load_data(SAMPLE_THREATS)
        threat_layout.addWidget(self.threat_table)
        threat_tab.setLayout(threat_layout)
        self.tabs.addTab(threat_tab, "Threats")

        # Telemetry Tab
        telemetry_tab = QWidget()
        telemetry_layout = QVBoxLayout()
        self.telemetry_viewer = TelemetryViewer()
        telemetry_layout.addWidget(self.telemetry_viewer)
        telemetry_tab.setLayout(telemetry_layout)
        self.tabs.addTab(telemetry_tab, "Telemetry")

        # Simulation Controls Tab
        sim_tab = QWidget()
        sim_layout = QVBoxLayout()
        self.sim_controls = SimulationControls()
        sim_layout.addWidget(self.sim_controls)
        sim_tab.setLayout(sim_layout)
        self.tabs.addTab(sim_tab, "Simulation Controls")

        # Reports Tab
        report_tab = QWidget()
        report_layout = QVBoxLayout()
        self.report_card = ReportCard(title="Daily Security Report", summary="No critical threats detected.")
        report_layout.addWidget(self.report_card)
        report_tab.setLayout(report_layout)
        self.tabs.addTab(report_tab, "Reports")

        # Attack Graph Tab
        graph_tab = QWidget()
        graph_layout = QVBoxLayout()
        self.attack_graph = AttackGraph()
        graph_layout.addWidget(self.attack_graph)
        graph_tab.setLayout(graph_layout)
        self.tabs.addTab(graph_tab, "Attack Graph")

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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SentenialXDashboard()
    window.show()
    sys.exit(app.exec())
