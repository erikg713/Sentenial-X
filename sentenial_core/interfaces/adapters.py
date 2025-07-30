# sentenial_core/interfaces/adapters.py

import abc
import asyncio
import json
from kafka import KafkaConsumer, KafkaProducer
import websockets
from loguru import logger


class StreamAdapter(abc.ABC):
    """
    Abstract base class for streaming adapters.
    Defines interface for connecting, consuming, and producing messages.
    """

    @abc.abstractmethod
    def connect(self):
        pass

    @abc.abstractmethod
    def disconnect(self):
        pass

    @abc.abstractmethod
    async def consume(self):
        """
        Async generator to yield incoming messages.
        """
        pass

    @abc.abstractmethod
    async def produce(self, message):
        """
        Send a message to the stream.
        """
        pass


class KafkaAdapter(StreamAdapter):
    def __init__(self, topic: str, bootstrap_servers: str = "localhost:9092", group_id: str = None):
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.consumer = None
        self.producer = None

    def connect(self):
        logger.info(f"Connecting Kafka consumer to topic {self.topic} at {self.bootstrap_servers}")
        self.consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset='latest',
            enable_auto_commit=True
        )
        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    def disconnect(self):
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer closed")
        if self.producer:
            self.producer.close()
            logger.info("Kafka producer closed")

    async def consume(self):
        if not self.consumer:
            raise RuntimeError("Kafka consumer is not connected")
        logger.info("Starting Kafka consumption loop")
        for message in self.consumer:
            yield message.value

    async def produce(self, message):
        if not self.producer:
            raise RuntimeError("Kafka producer is not connected")
        logger.debug(f"Producing message to Kafka topic {self.topic}: {message}")
        self.producer.send(self.topic, message)
        self.producer.flush()


class WebSocketAdapter(StreamAdapter):
    def __init__(self, uri: str):
        self.uri = uri
        self.ws = None

    async def connect(self):
        logger.info(f"Connecting WebSocket client to {self.uri}")
        self.ws = await websockets.connect(self.uri)

    async def disconnect(self):
        if self.ws:
            await self.ws.close()
            logger.info("WebSocket connection closed")

    async def consume(self):
        if not self.ws:
            raise RuntimeError("WebSocket is not connected")
        async for message in self.ws:
            yield json.loads(message)

    async def produce(self, message):
        if not self.ws:
            raise RuntimeError("WebSocket is not connected")
        await self.ws.send(json.dumps(message))
        logger.debug(f"Sent message to WebSocket {self.uri}: {message}")


class PiNetLogAdapter(StreamAdapter):
    """
    Adapter to tail Pi Network log files for real-time ingestion.
    Example implementation using asyncio to follow file changes.
    """
    def __init__(self, log_path: str):
        self.log_path = log_path
        self._file = None
        self._running = False

    def connect(self):
        self._file = open(self.log_path, "r")
        # Go to end of file to follow new entries
        self._file.seek(0, 2)
        self._running = True
        logger.info(f"Opened PiNet log file at {self.log_path} for streaming")

    def disconnect(self):
        if self._file:
            self._file.close()
            logger.info("Closed PiNet log file")

    async def consume(self):
        if not self._file:
            raise RuntimeError("Log file not opened")
        while self._running:
            line = self._file.readline()
            if not line:
                await asyncio.sleep(0.5)
                continue
            try:
                # Assuming JSON log lines; adjust as needed
                yield json.loads(line.strip())
            except Exception as e:
                logger.error(f"Failed to parse log line: {line.strip()} with error: {e}")

    async def produce(self, message):
        # PiNet logs are typically read-only
        raise NotImplementedError("PiNet log adapter does not support producing messages")


# You can extend with other adapters like REST API adapters, database adapters, etc.


