#!/usr/bin/env python3
# sentenial_x/core/cortex/cli.py
"""
Sentenial-X Cortex CLI

Provides a compact, robust command line interface to:
 - Train the NLP intent classifier
 - Run the real-time NLP stream processor (Kafka or WebSocket)

This module focuses on clear argument handling, validation, logging, and
graceful shutdown.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Optional

from .stream_handler import StreamHandler
from .model_trainer import train_model

LOGGER = logging.getLogger("sentenial_x.cortex.cli")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _validate_train_path(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        LOGGER.error("Training data file not found: %s", p)
        raise FileNotFoundError(p)
    if not p.is_file():
        LOGGER.error("Training data path is not a file: %s", p)
        raise IsADirectoryError(p)
    return p


def _validate_run_args(mode: str, kafka: Optional[str], topic: Optional[str], ws: Optional[str]) -> None:
    if mode == "kafka":
        if not kafka:
            raise ValueError("Kafka bootstrap server (--kafka) is required for mode 'kafka'.")
        if not topic:
            raise ValueError("Kafka topic (--topic) is required for mode 'kafka'.")
    elif mode == "websocket":
        if not ws:
            raise ValueError("WebSocket server URL (--ws) is required for mode 'websocket'.")
    else:
        raise ValueError("Unsupported mode: %s" % mode)


def _register_signal_handlers(stream: StreamHandler) -> None:
    def _handle_signal(signum, frame):
        LOGGER.info("Received signal %s, shutting down...", signum)
        try:
            stream.stop()
        except Exception as exc:  # pragmatic catch to ensure process exits
            LOGGER.debug("Exception while stopping stream: %s", exc)
        # Do not sys.exit() inside a signal handler; let main flow handle exit.
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)


def main(argv: Optional[list[str]] = None) -> int:
    """
    Entry point for the CLI. Returns an exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="🧠 Sentenial-X Cortex CLI - NLP Stream Processor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Examples:\n"
               "  train:   sentenial_x cortex train --data ./data/train.csv --verbose\n"
               "  run kafka: sentenial_x cortex run --mode kafka --kafka localhost:9092 --topic intent-events\n"
               "  run ws:    sentenial_x cortex run --mode websocket --ws ws://localhost:8765",
    )

    # Global args
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose (debug) logging")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run")

    # Train
    train_parser = subparsers.add_parser("train", help="Train the NLP intent classifier")
    train_parser.add_argument("--data", "-d", type=str, required=True, help="Path to training CSV")

    # Run Stream
    run_parser = subparsers.add_parser("run", help="Run real-time NLP processor")
    run_parser.add_argument("--mode", "-m", type=str, choices=["kafka", "websocket"], default="kafka", help="Stream mode")
    run_parser.add_argument("--topic", "-t", type=str, help="Kafka topic name")
    run_parser.add_argument("--kafka", type=str, help="Kafka bootstrap server (host:port)")
    run_parser.add_argument("--ws", type=str, help="WebSocket server URL (e.g. ws://localhost:8765)")

    args = parser.parse_args(argv)

    _setup_logging(args.verbose)
    LOGGER.debug("Parsed args: %s", args)

    try:
        if args.command == "train":
            data_path = _validate_train_path(args.data)
            LOGGER.info("Starting training using data: %s", data_path)
            # train_model may log its own progress; we capture exceptions here
            train_model(str(data_path))
            LOGGER.info("Training completed successfully.")
            return 0

        elif args.command == "run":
            # Validate run args
            try:
                _validate_run_args(args.mode, args.kafka, args.topic, args.ws)
            except ValueError as err:
                LOGGER.error("Invalid run configuration: %s", err)
                return 2

            LOGGER.info("Initializing StreamHandler (mode=%s)...", args.mode)
            stream = StreamHandler(
                mode=args.mode,
                kafka_topic=args.topic,
                kafka_bootstrap=args.kafka,
                ws_url=args.ws
            )

            # Register handlers so we can attempt a graceful shutdown
            try:
                _register_signal_handlers(stream)
            except Exception:
                LOGGER.debug("Signal handler registration failed or not supported on this platform.", exc_info=True)

            try:
                LOGGER.info("Starting stream processor.")
                stream.start()
                LOGGER.info("Stream processor exited normally.")
                return 0
            except KeyboardInterrupt:
                LOGGER.info("Interrupted by user, shutting down stream.")
                try:
                    stream.stop()
                except Exception:
                    LOGGER.debug("Error while stopping stream after KeyboardInterrupt.", exc_info=True)
                return 130
            except Exception as exc:
                LOGGER.exception("Unhandled exception while running stream: %s", exc)
                try:
                    stream.stop()
                except Exception:
                    LOGGER.debug("Error while stopping stream after exception.", exc_info=True)
                return 1

        else:
            # argparse with required subcommand should prevent reaching here, but guard anyway
            parser.print_help()
            return 0

    except FileNotFoundError:
        return 3
    except IsADirectoryError:
        return 4
    except Exception as exc:
        LOGGER.exception("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
