# sentenial_x/core/cortex/stream_handler.py

import asyncio
import json
from loguru import logger
from kafka import KafkaConsumer
from websockets import connect
from .nlp_engine import classify_intent

class StreamHandler:
    def __init__(self, mode="kafka", kafka_topic="pinet_logs", ws_url=None, kafka_bootstrap="localhost:9092"):
        self.mode = mode
        self.kafka_topic = kafka_topic
        self.kafka_bootstrap = kafka_bootstrap
        self.ws_url = ws_url

    def start(self):
        if self.mode == "kafka":
            self._start_kafka_stream()
        elif self.mode == "websocket":
            asyncio.run(self._start_ws_stream())
        else:
            logger.error(f"Unknown stream mode: {self.mode}")

    def _start_kafka_stream(self):
        consumer = KafkaConsumer(
            self.kafka_topic,
            bootstrap_servers=self.kafka_bootstrap,
            value_deserializer=lambda m: json.loads(m.decode("utf-8"))
        )
        logger.info(f"[CORTEX] Listening to Kafka topic: {self.kafka_topic}")
        for message in consumer:
            self.process_event(message.value)

    async def _start_ws_stream(self):
        logger.info(f"[CORTEX] Connecting to WebSocket: {self.ws_url}")
        async with connect(self.ws_url) as websocket:
            while True:
                message = await websocket.recv()
                self.process_event(json.loads(message))

    def process_event(self, data: dict):
        text = data.get("message") or data.get("log") or ""
        if text:
            logger.debug(f"[CORTEX] Received: {text}")
            label = classify_intent(text)
            logger.success(f"[CORTEX] Intent classified: {label} | Source: {data.get('source', 'unknown')}")


