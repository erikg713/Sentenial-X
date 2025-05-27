import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QTextEdit, QListWidget, QMessageBox
)
from PyQt5.QtCore import Qt

from sentenial_core.simulator.Wormgpt_clone import WormGPTClone

class WormGPTUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sentenial X A.I. - WormGPT Threat Simulator")
        self.setMinimumSize(700, 500)
        self.wormgpt = WormGPTClone()
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout()
        header = QLabel("WormGPT Threat Scenario Generator")
        header.setStyleSheet("font-size: 20px; font-weight: bold; margin: 10px 0;")
        main_layout.addWidget(header, alignment=Qt.AlignCenter)

        # Controls
        control_layout = QHBoxLayout()
        self.category_combo = QComboBox()
        self.category_combo.addItems(sorted(self.wormgpt._scenarios.keys()))
        control_layout.addWidget(QLabel("Scenario Category:"))
        control_layout.addWidget(self.category_combo)

        self.generate_btn = QPushButton("Generate Scenario")
        self.generate_btn.clicked.connect(self.generate_scenario)
        control_layout.addWidget(self.generate_btn)

        self.bulk_btn = QPushButton("Bulk Generate (5)")
        self.bulk_btn.clicked.connect(self.bulk_generate)
        control_layout.addWidget(self.bulk_btn)
        main_layout.addLayout(control_layout)

        # Scenario display
        self.scenario_list = QListWidget()
        self.scenario_list.setStyleSheet("font-family: monospace;")
        main_layout.addWidget(self.scenario_list, stretch=2)

        # Add scenario controls
        add_layout = QHBoxLayout()
        self.new_category = QComboBox()
        self.new_category.setEditable(True)
        self.new_category.addItems(sorted(self.wormgpt._scenarios.keys()))
        add_layout.addWidget(QLabel("Add/Select Category:"))
        add_layout.addWidget(self.new_category)

        self.new_scenario_text = QTextEdit()
        self.new_scenario_text.setPlaceholderText("Describe the new scenario here...")
        self.new_scenario_text.setFixedHeight(50)
        add_layout.addWidget(self.new_scenario_text)

        self.add_btn = QPushButton("Add Scenario")
        self.add_btn.clicked.connect(self.add_scenario)
        add_layout.addWidget(self.add_btn)
        main_layout.addLayout(add_layout)

        # Clear button
        self.clear_btn = QPushButton("Clear History")
        self.clear_btn.clicked.connect(self.clear_history)
        main_layout.addWidget(self.clear_btn, alignment=Qt.AlignRight)

        self.setLayout(main_layout)

    def generate_scenario(self):
        category = self.category_combo.currentText()
        scenario = self.wormgpt.generate_scenario(category)
        self.scenario_list.addItem(
            f"{scenario['timestamp']} | {scenario['category'].replace('_',' ').title()}: {scenario['description']}"
        )

    def bulk_generate(self):
        category = self.category_combo.currentText()
        scenarios = self.wormgpt.bulk_generate(5, category=category)
        for scenario in scenarios:
            self.scenario_list.addItem(
                f"{scenario['timestamp']} | {scenario['category'].replace('_',' ').title()}: {scenario['description']}"
            )

    def add_scenario(self):
        category = self.new_category.currentText().strip()
        description = self.new_scenario_text.toPlainText().strip()
        if not category or not description:
            QMessageBox.warning(self, "Missing Data", "Please provide both a category and a scenario description.")
            return
        self.wormgpt.add_scenario(category, description)
        if self.category_combo.findText(category) == -1:
            self.category_combo.addItem(category)
            self.new_category.addItem(category)
        QMessageBox.information(self, "Added", f"Scenario added to '{category}'.")
        self.new_scenario_text.clear()

    def clear_history(self):
        self.wormgpt.clear_history()
        self.scenario_list.clear()
        QMessageBox.information(self, "Cleared", "Scenario history cleared.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = WormGPTUI()
    ui.show()
    sys.exit(app.exec_())