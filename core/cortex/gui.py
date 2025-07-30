# sentenial_x/core/cortex/gui.py

import sys
import asyncio
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from .model_loader import CyberIntentModel

class Worker(QObject):
    result_ready = pyqtSignal(str, str)

    def __init__(self, model):
        super().__init__()
        self.model = model

    def classify_text(self, text):
        label = self.model.predict(text)
        self.result_ready.emit(text, label)

class CortexGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sentenial-X Cortex NLP Live Viewer")
        self.setGeometry(100, 100, 600, 400)

        self.model = CyberIntentModel()
        self.worker = Worker(self.model)

        layout = QVBoxLayout()

        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("Enter suspicious log message here...")
        layout.addWidget(self.input_text)

        self.classify_button = QPushButton("Classify")
        layout.addWidget(self.classify_button)

        self.result_list = QListWidget()
        layout.addWidget(self.result_list)

        self.classify_button.clicked.connect(self.on_classify)
        self.worker.result_ready.connect(self.display_result)

        self.setLayout(layout)

    def on_classify(self):
        text = self.input_text.toPlainText().strip()
        if not text:
            return
        # Run classification in a thread to avoid UI blocking
        QThread.create(lambda: self.worker.classify_text(text)).start()

    def display_result(self, text, label):
        item = QListWidgetItem(f"Text: {text}\nIntent: {label}")
        self.result_list.addItem(item)

def run_gui():
    app = QApplication(sys.argv)
    gui = CortexGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_gui()

