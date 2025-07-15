from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit

class ReportCard(QWidget):
    def __init__(self, title="Report Title", summary="Summary", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)

        layout = QVBoxLayout()

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        layout.addWidget(self.title_label)

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setText(summary)
        layout.addWidget(self.summary_text)

        self.setLayout(layout)
