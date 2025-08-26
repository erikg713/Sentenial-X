"""
Message queue helpers for Kafka and RabbitMQ.

This module provides small, well-documented wrappers around Kafka and RabbitMQ
clients to make connections more robust and easier to use across the codebase.

Goals / changes from the original:
- Add sensible defaults driven by environment variables.
- Add type hints and docstrings.
- Lazy-import external libraries to avoid import-time failures in environments
  where Kafka or RabbitMQ clients are not installed.
- Add retry/backoff for connections.
- Provide context-manager wrappers for automatic cleanup.
- Provide simple produce/consume convenience helpers and pluggable serializers.
- Improve logging and error messages.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import Callable, Generator, Iterable, Optional, Tuple

# Use the package logger if available; otherwise fallback to standard logging
try:
    from . import logger  # project's logger
except Exception:  # pragma: no cover - fallback for standalone use
    import logging

    logger = logging.getLogger(__name__)
    if not logger.handlers:
        # Basic default handler for safety in environments where package logger isn't configured
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# --- Utilities -----------------------------------------------------------------


def _retry(
    func: Callable,
    attempts: int = 3,
    initial_delay: float = 0.5,
    backoff: float = 2.0,
    allowed_exceptions: Tuple[type, ...] = (Exception,),
    name: Optional[str] = None,
):
    """
    Retry helper with exponential backoff. Returns the result of func() or
    raises the last exception after exhausting attempts.
    """
    name = name or getattr(func, "__name__", "operation")
    delay = initial_delay
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            logger.debug("Attempt %d/%d for %s", attempt, attempts, name)
            return func()
        except allowed_exceptions as exc:
            last_exc = exc
            logger.warning(
                "Attempt %d/%d failed for %s: %s", attempt, attempts, name, exc
            )
            if attempt == attempts:
                logger.error("All %d attempts failed for %s", attempts, name)
                raise
            time.sleep(delay)
            delay *= backoff
    # Should never reach here
    raise last_exc


# --- Kafka ---------------------------------------------------------------------


class KafkaUnavailableError(RuntimeError):
    pass


def _lazy_import_kafka():
    try:
        from kafka import KafkaProducer as _KafkaProducer, KafkaConsumer as _KafkaConsumer
        from kafka.errors import KafkaError  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive
        raise KafkaUnavailableError(
            "kafka-python package is not installed or failed to import"
        ) from exc
    return _KafkaProducer, _KafkaConsumer, KafkaError


def default_serializer(value: object) -> bytes:
    """Default serializer: JSON -> UTF-8 bytes. Accepts bytes passthrough."""
    if value is None:
        return b""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def default_deserializer(value: Optional[bytes]) -> object:
    if value is None:
        return None
    try:
        return json.loads(value.decode("utf-8"))
    except Exception:
        # Fall back to raw bytes if not JSON
        return value


class KafkaProducerWrapper:
    """
    Minimal wrapper around kafka.KafkaProducer with context manager,
    serializer support and optional retries on send errors.
    """

    def __init__(
        self,
        bootstrap_servers: Optional[str] = None,
        value_serializer: Callable[[object], bytes] = default_serializer,
        retries: int = 5,
        **producer_kwargs,
    ):
        self._KafkaProducer, _, _ = _lazy_import_kafka()
        self.bootstrap_servers = (
            bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        )
        self.value_serializer = value_serializer
        self.retries = retries
        logger.info("Creating Kafka producer for %s", self.bootstrap_servers)
        self._producer = self._KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=self.value_serializer,
            **producer_kwargs,
        )

    def send(self, topic: str, value: object, key: Optional[bytes] = None, **send_kwargs):
        """
        Send a message to Kafka. This returns the kafka.FutureRecordMetadata
        as returned by kafka-python. The send operation will be retried
        locally on failures up to `self.retries`.
        """
        def _op():
            return self._producer.send(topic, value=value, key=key, **send_kwargs)

        return _retry(_op, attempts=self.retries, name=f"kafka.send to {topic}")

    def flush(self, timeout: Optional[float] = None):
        logger.debug("Flushing Kafka producer (timeout=%s)", timeout)
        return self._producer.flush(timeout=timeout)

    def close(self, timeout: Optional[float] = None):
        logger.info("Closing Kafka producer")
        try:
            self.flush(timeout=timeout)
        finally:
            self._producer.close(timeout=timeout)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class KafkaConsumerWrapper:
    """
    Wrapper for kafka.KafkaConsumer that yields a generator of deserialized messages.
    """

    def __init__(
        self,
        topics: Iterable[str],
        bootstrap_servers: Optional[str] = None,
        group_id: str = "sentenial",
        value_deserializer: Callable[[Optional[bytes]], object] = default_deserializer,
        auto_offset_reset: str = "earliest",
        consumer_kwargs: Optional[dict] = None,
    ):
        _, KafkaConsumer, _ = _lazy_import_kafka()
        self.bootstrap_servers = (
            bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        )
        consumer_kwargs = consumer_kwargs or {}
        logger.info(
            "Creating Kafka consumer for topics=%s at %s (group=%s)",
            list(topics),
            self.bootstrap_servers,
            group_id,
        )
        self.consumer = KafkaConsumer(
            *topics,
            bootstrap_servers=self.bootstrap_servers,
            group_id=group_id,
            value_deserializer=value_deserializer,
            auto_offset_reset=auto_offset_reset,
            **consumer_kwargs,
        )

    def __iter__(self):
        return self.consumer

    def poll(self, timeout_ms: int = 1000, max_records: int = 500):
        """
        Poll the underlying consumer once and return records.
        """
        return self.consumer.poll(timeout_ms=timeout_ms, max_records=max_records)

    def close(self):
        logger.info("Closing Kafka consumer")
        try:
            self.consumer.close()
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Error while closing Kafka consumer: %s", exc)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# Convenience factory functions kept for backward compatibility
def kafka_producer(bootstrap_servers: Optional[str] = None, **kwargs) -> KafkaProducerWrapper:
    return KafkaProducerWrapper(bootstrap_servers=bootstrap_servers, **kwargs)


def kafka_consumer(topic: str, bootstrap_servers: Optional[str] = None, group_id: str = "sentenial", **kwargs) -> KafkaConsumerWrapper:
    # keep a simple interface for a single topic
    return KafkaConsumerWrapper(topics=[topic], bootstrap_servers=bootstrap_servers, group_id=group_id, **kwargs)


# --- RabbitMQ -----------------------------------------------------------------


class RabbitMQUnavailableError(RuntimeError):
    pass


def _lazy_import_pika():
    try:
        import pika  # type: ignore
    except Exception as exc:
        raise RabbitMQUnavailableError("pika package is not installed or failed to import") from exc
    return pika


class RabbitChannel:
    """
    Context-manager wrapper around pika.BlockingConnection and channel.

    Example:
        with RabbitChannel(host="localhost") as channel:
            channel.basic_publish(exchange="", routing_key="queue", body=b"hello")
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: int = 5672,
        virtual_host: str = "/",
        credentials: Optional[Tuple[str, str]] = None,
        heartbeat: int = 600,
        blocked_connection_timeout: int = 300,
        attempts: int = 3,
        retry_initial: float = 0.5,
        retry_backoff: float = 2.0,
    ):
        self.pika = _lazy_import_pika()
        self.host = host or os.getenv("RABBITMQ_HOST", "localhost")
        self.port = port
        self.virtual_host = virtual_host
        self.credentials = credentials or (
            os.getenv("RABBITMQ_USER", "guest"),
            os.getenv("RABBITMQ_PASSWORD", "guest"),
        )
        self.heartbeat = heartbeat
        self.blocked_connection_timeout = blocked_connection_timeout
        self.attempts = attempts
        self.retry_initial = retry_initial
        self.retry_backoff = retry_backoff
        self._connection = None
        self._channel = None

    def _connect(self):
        user, pwd = self.credentials
        creds = self.pika.PlainCredentials(user, pwd)
        params = self.pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            virtual_host=self.virtual_host,
            credentials=creds,
            heartbeat=self.heartbeat,
            blocked_connection_timeout=self.blocked_connection_timeout,
        )
        logger.info("Connecting to RabbitMQ at %s:%d vhost=%s", self.host, self.port, self.virtual_host)
        conn = self.pika.BlockingConnection(params)
        chan = conn.channel()
        return conn, chan

    def connect(self):
        def _attempt():
            conn, chan = self._connect()
            self._connection = conn
            self._channel = chan
            return chan

        return _retry(
            _attempt,
            attempts=self.attempts,
            initial_delay=self.retry_initial,
            backoff=self.retry_backoff,
            name=f"rabbitmq.connect to {self.host}:{self.port}",
        )

    @property
    def channel(self):
        if self._channel is None:
            return self.connect()
        return self._channel

    def close(self):
        logger.info("Closing RabbitMQ connection/channel")
        try:
            if self._channel and self._channel.is_open:
                try:
                    self._channel.close()
                except Exception:
                    # channel.close may raise on broken connections
                    logger.debug("Channel close raised an exception", exc_info=True)
            if self._connection and self._connection.is_open:
                try:
                    self._connection.close()
                except Exception:
                    logger.debug("Connection close raised an exception", exc_info=True)
        finally:
            self._channel = None
            self._connection = None

    def __enter__(self):
        self.connect()
        return self._channel

    def __exit__(self, exc_type, exc, tb):
        self.close()


def rabbitmq_channel(host: Optional[str] = None, **kwargs) -> RabbitChannel:
    """
    Return a RabbitChannel instance. Use it as a context manager:

        with rabbitmq_channel() as channel:
            channel.basic_publish(...)

    This preserves backward compatibility with the previous function name while
    providing a richer object.
    """
    return RabbitChannel(host=host, **kwargs)


# --- Exports ------------------------------------------------------------------


__all__ = [
    "KafkaProducerWrapper",
    "KafkaConsumerWrapper",
    "kafka_producer",
    "kafka_consumer",
    "RabbitChannel",
    "rabbitmq_channel",
                 ]
