# sentenial_x/core/cortex/gui.py

import sys
import asyncio
import json
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import QThread, pyqtSignal, QObject, QTimer
from kafka import KafkaConsumer
import websockets
from .model_loader import CyberIntentModel

class StreamWorker(QObject):
    new_log = pyqtSignal(str, str)  # text, intent label

    def __init__(self, model, mode="kafka", kafka_topic="pinet_logs", kafka_bootstrap="localhost:9092", ws_url=None):
        super().__init__()
        self.model = model
        self.mode = mode
        self.kafka_topic = kafka_topic
        self.kafka_bootstrap = kafka_bootstrap
        self.ws_url = ws_url
        self.running = False

    def start(self):
        self.running = True
        if self.mode == "kafka":
            self._consume_kafka()
        elif self.mode == "websocket":
            asyncio.run(self._consume_ws())
        else:
            print(f"Unknown mode: {self.mode}")

    def stop(self):
        self.running = False

    def _consume_kafka(self):
        consumer = KafkaConsumer(
            self.kafka_topic,
            bootstrap_servers=self.kafka_bootstrap,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset='latest'
        )
        for message in consumer:
            if not self.running:
                break
            text = message.value.get("message", "") or message.value.get("log", "")
            if text:
                label = self.model.predict(text)
                self.new_log.emit(text, label)

    async def _consume_ws(self):
        async with websockets.connect(self.ws_url) as ws:
            while self.running:
                msg = await ws.recv()
                data = json.loads(msg)
                text = data.get("message", "") or data.get("log", "")
                if text:
                    label = self.model.predict(text)
                    self.new_log.emit(text, label)

class CortexGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sentenial-X Cortex NLP Live Viewer")
        self.setGeometry(100, 100, 700, 500)

        self.model = CyberIntentModel()
        self.stream_worker = StreamWorker(self.model, mode="kafka")  # or websocket
        self.thread = QThread()
        self.stream_worker.moveToThread(self.thread)
        self.thread.started.connect(self.stream_worker.start)

        self.stream_worker.new_log.connect(self.display_result)

        layout = QVBoxLayout()

        self.result_list = QListWidget()
        layout.addWidget(self.result_list)

        self.start_button = QPushButton("Start Stream")
        self.stop_button = QPushButton("Stop Stream")
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        self.start_button.clicked.connect(self.start_stream)
        self.stop_button.clicked.connect(self.stop_stream)

        self.setLayout(layout)

    def start_stream(self):
        if not self.thread.isRunning():
            self.thread.start()

    def stop_stream(self):
        self.stream_worker.stop()
        self.thread.quit()
        self.thread.wait()

    def display_result(self, text, label):
        item = QListWidgetItem(f"Text: {text}\nIntent: {label}")
        self.result_list.addItem(item)
        self.result_list.scrollToBottom()

def run_gui():
    app = QApplication(sys.argv)
    gui = CortexGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_gui()
