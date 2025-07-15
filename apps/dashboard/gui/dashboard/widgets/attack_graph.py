from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGraphicsView, QGraphicsScene
from PySide6.QtGui import QBrush, QPen, QColor
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsLineItem

class AttackGraph(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Attack Graph")
        self.resize(700, 500)
        layout = QVBoxLayout()

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        layout.addWidget(self.view)

        # Example nodes and edges
        node1 = QGraphicsEllipseItem(0, 0, 30, 30)
        node1.setBrush(QBrush(QColor("skyblue")))
        node1.setToolTip("Node 1: Initial Access")

        node2 = QGraphicsEllipseItem(100, 100, 30, 30)
        node2.setBrush(QBrush(QColor("lightgreen")))
        node2.setToolTip("Node 2: Privilege Escalation")

        edge = QGraphicsLineItem(15, 15, 115, 115)
        edge.setPen(QPen(Qt.black, 2))

        self.scene.addItem(edge)
        self.scene.addItem(node1)
        self.scene.addItem(node2)

        instructions = QLabel("Click nodes to inspect attack graph details.")
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setStyleSheet("font-style: italic; margin: 5px;")
        layout.addWidget(instructions)

        self.setLayout(layout)
