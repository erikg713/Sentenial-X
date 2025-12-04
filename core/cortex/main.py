#!/usr/bin/env python3
"""
Entry-point for the Cortex stream demo / runner.

Improvements:
- Structured logging with configurable level.
- Graceful shutdown with cancel/wait of the stream task.
- Small CLI to run demo mode or keep the stream running.
- Clear async helpers and type annotations.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from typing import Any, Dict, Iterable, List, Optional

from sentenial_x.core.cortex import Brainstem, DecisionEngine, SemanticAnalyzer, SignalRouter
from sentenial_x.core.cortex.stream_processor import StreamProcessor

LOG = logging.getLogger("sentenial_x.cortex.main")

# Small set of demo signals used when running in demo mode
SAMPLE_SIGNALS: List[Dict[str, Any]] = [
    {"id": "s1", "threat_level": 9, "description": "Detected RCE payload targeting Apache"},
    {"id": "s2", "threat_level": 4, "description": "Unusual process tree with encoded powershell"},
    {"id": "s3", "threat_level": 2, "description": "User logged in from new device"},
]


def configure_logging() -> None:
    """
    Configure root logger. Honors environment variable SENTENIAL_DEBUG to set DEBUG level.
    """
    level = logging.DEBUG if os.getenv("SENTENIAL_DEBUG", "").lower() in ("1", "true", "yes") else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s %(levelname)-5s [%(name)s] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers if configure_logging is called multiple times
    if not any(isinstance(h, logging.StreamHandler) and h.stream is sys.stdout for h in root.handlers):
        root.addHandler(handler)


async def _run_stream_guard(stream: StreamProcessor) -> None:
    """
    Run the stream processor and guard against unexpected exceptions so they are logged.
    This coroutine is intended to be executed as a background task.
    """
    try:
        LOG.info("Starting StreamProcessor main loop")
        await stream.start_stream()
        LOG.info("StreamProcessor finished normally")
    except asyncio.CancelledError:
        LOG.debug("StreamProcessor task was cancelled")
        raise
    except Exception:
        LOG.exception("Unhandled exception in StreamProcessor task")
        # Depending on desired semantics we could re-raise to crash the app; here we just log.
        raise


async def feed_signals(stream: StreamProcessor, signals: Iterable[Dict[str, Any]], delay: float = 0.1) -> int:
    """
    Feed signals into the stream processor with an optional small delay between them.

    Returns the number of signals successfully submitted.
    """
    submitted = 0
    for sig in signals:
        try:
            await stream.add_signal(sig)
            submitted += 1
            LOG.debug("Submitted signal: %s", sig.get("id", "<no-id>"))
        except Exception:
            LOG.exception("Failed to submit signal: %s", sig)
        if delay:
            await asyncio.sleep(delay)
    LOG.info("Finished feeding %d signals", submitted)
    return submitted


async def shutdown_stream(stream: StreamProcessor, task: asyncio.Task, timeout: float = 5.0) -> None:
    """
    Signal the StreamProcessor to stop and wait for the background task to finish.
    """
    LOG.info("Initiating stream shutdown")
    try:
        # ask the processor to stop accepting/processing new signals
        stream.stop_stream()
    except Exception:
        LOG.exception("Error while signaling StreamProcessor to stop")

    try:
        await asyncio.wait_for(task, timeout=timeout)
        LOG.info("StreamProcessor task stopped cleanly")
    except asyncio.TimeoutError:
        LOG.warning("StreamProcessor did not stop within %.1fs; cancelling task", timeout)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            LOG.debug("StreamProcessor task cancelled successfully")
        except Exception:
            LOG.exception("Exception while cancelling StreamProcessor task")


async def main(demo: bool = True, demo_signals: Optional[Iterable[Dict[str, Any]]] = None) -> None:
    """
    Main entrypoint for running the cortex stream.

    If demo=True, demo_signals are injected and the stream is shutdown shortly after.
    Otherwise, the process keeps running until interrupted (SIGINT/SIGTERM).
    """
    configure_logging()
    LOG.info("Initializing Cortex components")

    # Instantiate core components; this keeps the wiring explicit and easy to modify.
    brainstem = Brainstem()
    analyzer = SemanticAnalyzer()
    engine = DecisionEngine()
    router = SignalRouter(brainstem, analyzer, engine)

    # Create the stream processor and run it in a background task.
    stream = StreamProcessor(router)
    stream_task = asyncio.create_task(_run_stream_guard(stream), name="stream-processor")

    # Setup a stop event and hook OS signals (best-effort; add_signal_handler may not be available on Windows)
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _on_signal(signame: str) -> None:
        LOG.info("Received signal %s, scheduling shutdown", signame)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig: _on_signal(s.name))
        except NotImplementedError:
            LOG.debug("Loop does not support add_signal_handler (%s)", sig)

    try:
        # If demo mode, feed demo signals and stop after a short period
        if demo:
            LOG.info("Running in demo mode (automated signals will be injected)")
            signals = list(demo_signals or SAMPLE_SIGNALS)
            await feed_signals(stream, signals, delay=0.1)

            # allow some time for processing then shutdown
            await asyncio.sleep(2)
            await shutdown_stream(stream, stream_task)
        else:
            LOG.info("Running in continuous mode. Send SIGINT/SIGTERM to stop.")
            # Wait until a signal triggers stop_event
            await stop_event.wait()
            await shutdown_stream(stream, stream_task)
    except Exception:
        LOG.exception("Top-level error in main")
        # Ensure we attempt to shutdown cleanly if possible
        if not stream_task.done():
            await shutdown_stream(stream, stream_task)
        raise


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="cortex-main", description="Run the Sentenial-X Cortex stream processor.")
    p.add_argument("--demo", action="store_true", help="Run a short demo that injects sample signals and exits.")
    p.add_argument("--signals-file", type=str, help="Path to a JSON file containing a list of signals to load for demo mode.")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    demo_signals: Optional[List[Dict[str, Any]]] = None

    if args.signals_file:
        try:
            with open(args.signals_file, "r", encoding="utf-8") as fh:
                demo_signals = json.load(fh)
                if not isinstance(demo_signals, list):
                    LOG.warning("signals-file does not contain a list; ignoring")
                    demo_signals = None
        except Exception:
            # configure logging early so we can emit the error
            configure_logging()
            LOG.exception("Failed to read signals file %s; running without it", args.signals_file)

    # Run the main coroutine and allow KeyboardInterrupt to terminate cleanly
    try:
        asyncio.run(main(demo=args.demo or bool(demo_signals), demo_signals=demo_signals))
    except KeyboardInterrupt:
        # This should be rare since signal handlers set stop_event; still keep safety net.
        LOG.info("Interrupted by user, exiting")
        sys.exit(0)
