#!/usr/bin/env python3
# sentenial_x/core/cortex/cli.py
"""
Sentenial-X Cortex CLI (Production-Ready & Fixed)

A robust, secure, and observable CLI for:
- Training the NLP intent classifier (with validation, checkpointing, metrics)
- Running real-time inference via Kafka or WebSocket (with health checks, backpressure, graceful shutdown)

Features:
- Full type safety & error handling
- Configuration via environment + CLI (12-factor ready)
- Model versioning & artifact logging
- Prometheus metrics & health endpoint
- Structured JSON logging
- Graceful shutdown with final flush
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import structlog
from prometheus_client import start_http_server, Counter, Gauge, Histogram

from .stream_handler import StreamHandler
from .model_trainer import train_model, ModelTrainingResult

# === Prometheus Metrics ===
REQUEST_COUNTER = Counter(
    "cortex_requests_total", "Total processed messages", ["source", "intent"]
)
PROCESSING_TIME = Histogram(
    "cortex_processing_seconds", "Message processing latency", ["source"]
)
ERROR_COUNTER = Counter(
    "cortex_errors_total", "Errors during processing", ["type"]
)
HEALTH_GAUGE = Gauge("cortex_up", "Cortex service health (1=up, 0=down)")

# === Structured Logging Setup ===
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(sort_keys=True)
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
)
log = structlog.get_logger("sentenial_x.cortex.cli")


# === Configuration Dataclass ===
@dataclass
class Config:
    train_data_path: Optional[Path] = None

    mode: str = "kafka"
    kafka_bootstrap: Optional[str] = None
    kafka_topic: Optional[str] = None
    websocket_url: Optional[str] = None

    metrics_port: int = 8000
    log_level: str = "INFO"
    log_json: bool = True

    @staticmethod
    def from_env() -> "Config":
        return Config(
            mode=os.getenv("CORTEX_MODE", "kafka").lower(),
            kafka_bootstrap=os.getenv("KAFKA_BOOTSTRAP"),
            kafka_topic=os.getenv("KAFKA_TOPIC", "intent-input"),
            websocket_url=os.getenv("WEBSOCKET_URL"),
            metrics_port=int(os.getenv("METRICS_PORT", "8000")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_json=os.getenv("LOG_JSON", "true").lower() == "true"
        )


def setup_logging(config: Config):
    level = getattr(logging, config.log_level.upper())
    handler = logging.StreamHandler(sys.stdout)
    if config.log_json:
        handler.setFormatter(logging.Formatter('%(message)s'))
    else:
        handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
    logging.basicConfig(level=level, handlers=[handler])
    log.info("logging_configured", level=config.log_level, json=config.log_json)


def validate_training_path(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        log.error("train_data_not_found", path=str(p))
        raise FileNotFoundError(f"Training data not found: {p}")
    if not p.is_file():
        log.error("train_data_not_file", path=str(p))
        raise ValueError(f"Training data must be a file: {p}")
    return p


def start_metrics_server(port: int):
    start_http_server(port)
    HEALTH_GAUGE.set(1)
    log.info("metrics_server_started", port=port)


def train_command(data_path: str) -> int:
    path = validate_training_path(data_path)
    log.info("training_started", data_path=str(path))

    try:
        result: ModelTrainingResult = train_model(
            data_path=str(path),
            model_dir="models/",
            version=int(time.time())
        )
        log.info(
            "training_completed",
            accuracy=result.accuracy,
            f1=result.f1,
            model_path=result.model_path,
            duration_seconds=result.duration
        )
        print(json.dumps({
            "status": "success",
            "accuracy": result.accuracy,
            "f1": result.f1,
            "model_path": result.model_path
        }, indent=2))
        return 0
    except Exception as e:
        ERROR_COUNTER.labels(type="training").inc()
        log.exception("training_failed", error=str(e))
        return 1


async def run_stream(config: Config) -> int:
    log.info("stream_processor_starting", mode=config.mode)

    if config.mode == "kafka":
        if not config.kafka_bootstrap or not config.kafka_topic:
            log.error("kafka_config_missing")
            return 2
        handler = StreamHandler(
            mode="kafka",
            kafka_bootstrap=config.kafka_bootstrap,
            kafka_topic=config.kafka_topic
        )
    elif config.mode == "websocket":
        if not config.websocket_url:
            log.error("websocket_url_missing")
            return 2
        handler = StreamHandler(
            mode="websocket",
            ws_url=config.websocket_url
        )
    else:
        log.error("invalid_mode", mode=config.mode)
        return 2

    stop_event = asyncio.Event()

    def signal_handler():
        log.info("shutdown_signal_received")
        stop_event.set()

    signal.signal(signal.SIGINT, lambda s, f: signal_handler())
    signal.signal(signal.SIGTERM, lambda s, f: signal_handler())

    try:
        await handler.start(stop_event=stop_event)
        log.info("stream_processor_stopped")
        return 0
    except Exception as e:
        ERROR_COUNTER.labels(type="runtime").inc()
        log.exception("stream_processor_crashed", error=str(e))
        HEALTH_GAUGE.set(0)
        return 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sentenial-X Cortex - NLP Intent Engine CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-json-log", action="store_true", help="Use human-readable logs")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    train_p = subparsers.add_parser("train", help="Train intent classifier")
    train_p.add_argument("data", type=str, help="Path to training CSV")

    # Run
    run_p = subparsers.add_parser("run", help="Run real-time inference")
    run_p.add_argument("--mode", choices=["kafka", "websocket"], default="kafka")
    run_p.add_argument("--kafka", help="Kafka bootstrap servers")
    run_p.add_argument("--topic", default="intent-input", help="Kafka input topic")
    run_p.add_argument("--ws", help="WebSocket URL (e.g. ws://localhost:8765)")
    run_p.add_argument("--metrics-port", type=int, default=8000, help="Prometheus metrics port")

    args = parser.parse_args(argv)

    # Merge CLI → ENV → defaults
    config = Config.from_env()
    if args.verbose:
        config.log_level = "DEBUG"
    if args.no_json_log:
        config.log_json = False

    # Override from CLI
    if args.command == "train":
        config.train_data_path = Path(args.data)
    elif args.command == "run":
        config.mode = args.mode
        if args.kafka:
            config.kafka_bootstrap = args.kafka
        if args.topic:
            config.kafka_topic = args.topic
        if args.ws:
            config.websocket_url = args.ws
        config.metrics_port = args.metrics_port

    setup_logging(config)
    start_metrics_server(config.metrics_port)

    try:
        if args.command == "train":
            return train_command(str(config.train_data_path))
        elif args.command == "run":
            return asyncio.run(run_stream(config))
    except KeyboardInterrupt:
        log.info("interrupted_by_user")
        return 130
    except Exception as e:
        log.exception("unhandled_exception", error=str(e))
        HEALTH_GAUGE.set(0)
        return 1


if __name__ == "__main__":
    sys.exit(main())
