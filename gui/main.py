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
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())