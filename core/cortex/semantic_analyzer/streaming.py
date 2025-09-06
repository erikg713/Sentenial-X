"""
core/cortex/semantic_analyzer/streaming.py

Improved real-time semantic analyzer streaming engine.

Key improvements:
- Proper async worker model with configurable concurrency.
- Graceful startup/shutdown and queue draining.
- Support for both sync and async analyzers/subscribers.
- Per-analyzer timeout and robust exception handling.
- Backpressure support on ingest (timeout option).
- Replaced prints with logging, added simple metrics counters.
- Async context manager support for cleaner resource handling.
- Small performance-minded tweaks (batch processing option).
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Union

# Import local analyzers here (kept same names as original)
from .scoring import score_evasion
from .anomaly import detect_anomaly

Logger = logging.getLogger(__name__)

# Types
Event = Dict[str, Any]
SyncAnalyzer = Callable[[Event], Event]
AsyncAnalyzer = Callable[[Event], Awaitable[Event]]
Analyzer = Union[SyncAnalyzer, AsyncAnalyzer]

SyncSubscriber = Callable[[Event], None]
AsyncSubscriber = Callable[[Event], Awaitable[None]]
Subscriber = Union[SyncSubscriber, AsyncSubscriber]


class StreamingEngine:
    """
    Async engine for handling streaming telemetry and semantic analysis.

    Usage:
        engine = StreamingEngine(concurrency=4)
        engine.add_analyzer(...)
        engine.subscribe(...)
        await engine.start()
        await engine.ingest({...})
        await engine.stop()

    Or as an async context manager:
        async with StreamingEngine() as engine:
            ...
    """

    def __init__(
        self,
        queue_size: int = 10_000,
        concurrency: int = 2,
        analyzer_timeout: float = 1.0,
        ingest_timeout: Optional[float] = 1.0,
        batch_size: int = 1,
    ):
        self.queue: asyncio.Queue[Optional[Event]] = asyncio.Queue(maxsize=queue_size)
        self._analyzers: List[Analyzer] = [self._apply_scoring, self._apply_anomaly]
        self._subscribers: List[Subscriber] = []
        self._workers: List[asyncio.Task] = []
        self._concurrency = max(1, int(concurrency))
        self._analyzer_timeout = float(analyzer_timeout)
        self._ingest_timeout = ingest_timeout  # None means block forever
        self._batch_size = max(1, int(batch_size))

        self.running = False

        # Simple metrics
        self.metrics = {
            "ingested": 0,
            "processed": 0,
            "failed_analyzers": 0,
            "subscriber_errors": 0,
        }

    # -------------------------
    # Public API
    # -------------------------
    async def start(self) -> None:
        """Start processing workers."""
        if self.running:
            Logger.debug("StreamingEngine already running.")
            return

        Logger.info("StreamingEngine starting with concurrency=%d, batch_size=%d", self._concurrency, self._batch_size)
        self.running = True
        self._workers = [asyncio.create_task(self._worker(i)) for i in range(self._concurrency)]

    async def stop(self, drain: bool = True) -> None:
        """
        Stop the engine.

        If drain is True (default) the queue will be processed until empty before stopping.
        Otherwise, workers are cancelled immediately.
        """
        if not self.running:
            Logger.debug("StreamingEngine already stopped.")
            return

        Logger.info("StreamingEngine stopping (drain=%s)...", drain)
        if drain:
            # Wait until queue empty
            while not self.queue.empty():
                await asyncio.sleep(0.05)

        # Signal workers to exit by putting sentinel None for each worker
        for _ in self._workers:
            await self.queue.put(None)

        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []
        self.running = False
        Logger.info("StreamingEngine stopped.")

    async def __aenter__(self) -> "StreamingEngine":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop(drain=True)

    async def ingest(self, event: Event) -> None:
        """
        Push an event into the queue.

        Will wait up to ingest_timeout seconds if queue is full (unless ingest_timeout is None, then block).
        Raises asyncio.TimeoutError on timeout.
        """
        if not self.running:
            # Allow ingest even if engine not started; queue will accumulate.
            Logger.debug("Ingesting while engine not running (queueing): %s", event)

        put_coro = self.queue.put(event)
        if self._ingest_timeout is None:
            await put_coro
        else:
            await asyncio.wait_for(put_coro, timeout=self._ingest_timeout)

        self.metrics["ingested"] += 1

    def add_analyzer(self, analyzer: Analyzer) -> None:
        """Register a new analyzer function (sync or async)."""
        if not callable(analyzer):
            raise TypeError("analyzer must be callable")
        self._analyzers.append(analyzer)
        Logger.debug("Analyzer added: %s", getattr(analyzer, "__name__", repr(analyzer)))

    def remove_analyzer(self, analyzer: Analyzer) -> None:
        """Remove analyzer if present."""
        try:
            self._analyzers.remove(analyzer)
            Logger.debug("Analyzer removed: %s", getattr(analyzer, "__name__", repr(analyzer)))
        except ValueError:
            Logger.warning("Tried to remove analyzer that was not registered: %s", analyzer)

    def subscribe(self, callback: Subscriber) -> None:
        """Subscribe to processed events (supports sync or async callbacks)."""
        if not callable(callback):
            raise TypeError("subscriber must be callable")
        self._subscribers.append(callback)
        Logger.debug("Subscriber added: %s", getattr(callback, "__name__", repr(callback)))

    def unsubscribe(self, callback: Subscriber) -> None:
        """Unsubscribe a previously registered callback."""
        try:
            self._subscribers.remove(callback)
            Logger.debug("Subscriber removed: %s", getattr(callback, "__name__", repr(callback)))
        except ValueError:
            Logger.warning("Tried to remove subscriber that was not registered: %s", callback)

    # -------------------------
    # Internal processing
    # -------------------------
    async def _worker(self, worker_id: int) -> None:
        """Worker loop: takes events from queue and processes them."""
        Logger.info("Worker[%d] started.", worker_id)
        try:
            while True:
                # Batch fetch to reduce queue overhead if requested
                batch: List[Event] = []
                item = await self.queue.get()
                if item is None:
                    # Sentinels are used to shut down workers
                    Logger.debug("Worker[%d] received stop sentinel.", worker_id)
                    self.queue.task_done()
                    break

                batch.append(item)
                # Try to pull up to batch_size-1 more items without waiting
                for _ in range(self._batch_size - 1):
                    try:
                        next_item = self.queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if next_item is None:
                        # Re-queue sentinel for other workers and break
                        await self.queue.put(None)
                        self.queue.task_done()
                        break
                    batch.append(next_item)

                # Process batch
                for event in batch:
                    try:
                        await self._process_event(event)
                        self.metrics["processed"] += 1
                    except Exception as ex:
                        Logger.exception("Unhandled error processing event: %s", ex)
                    finally:
                        # mark all queue items as done
                        self.queue.task_done()
        except asyncio.CancelledError:
            Logger.info("Worker[%d] cancelled.", worker_id)
            raise
        finally:
            Logger.info("Worker[%d] exiting.", worker_id)

    async def _process_event(self, event: Event) -> None:
        """
        Run analyzers in order and forward event to subscribers.

        A shallow copy of the event is made so analyzers can't accidentally mutate the original
        caller-owned dict (but they can still return a modified dict).
        """
        # Work on a shallow copy to avoid surprising external mutation
        current = dict(event)

        for analyzer in list(self._analyzers):
            try:
                if asyncio.iscoroutinefunction(analyzer):
                    # run analyzer with timeout
                    current = await asyncio.wait_for(analyzer(current), timeout=self._analyzer_timeout)
                else:
                    # run sync analyzer in the event loop (blocking small CPU bound functions are acceptable)
                    # wrap in timeout by running in a thread pool to avoid blocking the event loop
                    loop = asyncio.get_running_loop()
                    current = await asyncio.wait_for(loop.run_in_executor(None, analyzer, current), timeout=self._analyzer_timeout)
            except asyncio.TimeoutError:
                Logger.warning("Analyzer timed out: %s", getattr(analyzer, "__name__", repr(analyzer)))
                self.metrics["failed_analyzers"] += 1
            except Exception:
                Logger.exception("Analyzer raised exception: %s", getattr(analyzer, "__name__", repr(analyzer)))
                self.metrics["failed_analyzers"] += 1

            # Ensure analyzer returned a dict-like structure; if not, keep current
            if not isinstance(current, dict):
                Logger.warning("Analyzer did not return a dict; ignoring result from %s", getattr(analyzer, "__name__", repr(analyzer)))
                current = dict(current) if isinstance(current, dict) else dict(event)

        # Dispatch to subscribers (allowing sync or async)
        await self._dispatch_to_subscribers(current)

    async def _dispatch_to_subscribers(self, event: Event) -> None:
        """Send processed event to all subscribers; errors are logged per subscriber."""
        for sub in list(self._subscribers):
            try:
                if asyncio.iscoroutinefunction(sub):
                    await sub(event)
                else:
                    # run sync subscriber in executor to avoid blocking loop
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, sub, event)
            except Exception:
                Logger.exception("Subscriber error: %s", getattr(sub, "__name__", repr(sub)))
                self.metrics["subscriber_errors"] += 1


    # -------------------------
    # Built-in analyzers
    # -------------------------
    async def _apply_scoring(self, event: Event) -> Event:
        """Apply stealth scoring (async wrapper around the score_evasion function)."""
        try:
            # score_evasion may be sync; run in executor for safety if it's CPU-bound
            loop = asyncio.get_running_loop()
            score = await loop.run_in_executor(None, score_evasion, event)
            # do not mutate original input
            out = dict(event)
            out["evasion_score"] = score
            return out
        except Exception:
            Logger.exception("Error during scoring")
            return event

    async def _apply_anomaly(self, event: Event) -> Event:
        """Apply anomaly detection (async wrapper around detect_anomaly)."""
        try:
            loop = asyncio.get_running_loop()
            anomaly = await loop.run_in_executor(None, detect_anomaly, event)
            out = dict(event)
            out["anomaly"] = anomaly
            return out
        except Exception:
            Logger.exception("Error during anomaly detection")
            return event


# --- Example subscriber (WebSocket broadcaster) ---
async def websocket_broadcaster(event: Event) -> None:
    """Broadcasts event to WebSocket clients (stub, integrate with dashboard)."""
    # In real integration, replace with actual websocket send calls, e.g.:
    # await websocket_pool.broadcast(json.dumps(event))
    Logger.info("[WebSocket] Broadcast: %s", event)


# --- Example usage when run directly ---
if __name__ == "__main__":
    import json
    import random

    logging.basicConfig(level=logging.DEBUG)

    async def main():
        # small demo
        engine = StreamingEngine(concurrency=2, analyzer_timeout=0.5, ingest_timeout=0.5, batch_size=2)
        engine.subscribe(websocket_broadcaster)

        await engine.start()

        demo_events = [
            {"command": "powershell -ep bypass", "id": 1},
            {"command": "vssadmin delete shadows", "id": 2},
            {"command": "normal benign command", "id": 3},
        ]

        # ingest events with slight jitter
        for ev in demo_events:
            try:
                await engine.ingest(ev)
            except asyncio.TimeoutError:
                Logger.warning("Timed out while ingesting event: %s", ev)

        # allow some time for processing
        await asyncio.sleep(1.0)
        await engine.stop(drain=True)

        Logger.info("Metrics: %s", json.dumps(engine.metrics))

    asyncio.run(main())        event["evasion_score"] = score_evasion(event)
        return event

    async def _apply_anomaly(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Apply anomaly detection."""
        event["anomaly"] = detect_anomaly(event)
        return event

    def add_analyzer(self, analyzer: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """Register a new analyzer function."""
        self.analyzers.append(analyzer)

    def subscribe(self, callback: Callable[[Dict[str, Any]], Coroutine]):
        """Subscribe to processed events (e.g., dashboard broadcaster)."""
        self.subscribers.append(callback)

    async def _process_event(self, event: Dict[str, Any]):
        """Run analyzers and forward event to subscribers."""
        for analyzer in self.analyzers:
            if asyncio.iscoroutinefunction(analyzer):
                event = await analyzer(event)
            else:
                event = analyzer(event)

        for sub in self.subscribers:
            try:
                await sub(event)
            except Exception as e:
                print(f"[StreamingEngine] Subscriber error: {e}")

    async def run(self):
        """Main loop for processing events from queue."""
        self.running = True
        print("[StreamingEngine] Started event loop...")
        while self.running:
            event = await self.queue.get()
            await self._process_event(event)

    async def stop(self):
        """Stop the engine."""
        self.running = False
        print("[StreamingEngine] Stopping event loop...")


# --- Example subscriber (WebSocket broadcaster) ---

async def websocket_broadcaster(event: Dict[str, Any]):
    """Broadcasts event to WebSocket clients (stub, integrate with dashboard)."""
    print(f"[WebSocket] Broadcast: {json.dumps(event)}")


# --- Example usage ---
if __name__ == "__main__":
    async def main():
        engine = StreamingEngine()
        engine.subscribe(websocket_broadcaster)

        # Start engine loop
        asyncio.create_task(engine.run())

        # Ingest test events
        await engine.ingest({"command": "powershell -ep bypass"})
        await engine.ingest({"command": "vssadmin delete shadows"})
        await engine.ingest({"command": "normal benign command"})

        await asyncio.sleep(2)
        await engine.stop()

    asyncio.run(main())
