# gui/main.py
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget
from core import some_function  # adjust to match your core import

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Centennial X")

        self.text = QTextEdit()
        self.text.setReadOnly(True)

        btn = QPushButton("Run Core Function")
        btn.clicked.connect(self.run_core)

        layout = QVBoxLayout()
        layout.addWidget(btn)
        layout.addWidget(self.text)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def run_core(self):
        try:
            result = some_function()
        except Exception as e:
            self.text.setText(f"Error: {e}")
        else:
            self.text.setText(str(result))


if __name__ == "__main__":
    from sentenial_x.core.cortex import Brainstem, SemanticAnalyzer, DecisionEngine, SignalRouter

    # Instantiate cortex modules
    brainstem = Brainstem()
    analyzer = SemanticAnalyzer()
    engine = DecisionEngine()
    router = SignalRouter(brainstem, analyzer, engine)

    # Simulated signal
    incoming_signal = {
        "id": "signal-001",
        "threat_level": 9,
        "description": "Unauthorized escalation to root access using known CVE RCE payload."
    }

    result = router.handle(incoming_signal)
    print(result)
