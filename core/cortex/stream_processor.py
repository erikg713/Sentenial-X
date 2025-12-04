import asyncio
import inspect
import logging
import json
from typing import Any, Callable, Coroutine, Dict, Optional

_LOG = logging.getLogger(__name__)

_Sentinel = object()


class StreamProcessor:
    """
    Asynchronous stream processor that receives 'signals' and routes them via the provided router.

    Features:
    - Handles both sync and async router.handle(...) implementations.
    - Configurable worker concurrency and queue maxsize for backpressure.
    - Optional result_handler callback to customize result consumption.
    - Graceful start/stop with awaiting of in-flight work.
    - Metrics counters for basic observability.
    """

    def __init__(
        self,
        router: Any,
        *,
        queue_maxsize: int = 0,
        workers: int = 1,
        result_handler: Optional[Callable[[Any], Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.router = router
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=queue_maxsize)
        self.workers = max(1, int(workers))
        self._tasks: list[asyncio.Task] = []
        self._running = asyncio.Event()
        self._result_handler = result_handler
        self.logger = logger or _LOG

        # Basic metrics
        self.processed: int = 0
        self.errors: int = 0

    # ---- Public API ----

    async def add_signal(self, signal: Any, timeout: Optional[float] = None) -> None:
        """
        Put a signal onto the internal queue.

        - If timeout is provided, raises asyncio.TimeoutError on timeout.
        - If queue has maxsize and is full, this waits (subject to timeout) rather than dropping.
        """
        try:
            if timeout is None:
                await self.queue.put(signal)
            else:
                await asyncio.wait_for(self.queue.put(signal), timeout=timeout)
        except asyncio.TimeoutError:
            self.logger.warning("Timed out while adding signal to queue (backpressure active).")
            raise

    async def start(self) -> None:
        """
        Start background worker tasks. Safe to call multiple times (idempotent).
        """
        if self._running.is_set():
            self.logger.debug("StreamProcessor already running.")
            return

        self.logger.info("Starting StreamProcessor: workers=%d queue_maxsize=%d", self.workers, self.queue.maxsize)
        self._running.set()
        self._tasks = [asyncio.create_task(self._worker_loop(i)) for i in range(self.workers)]

    async def stop(self, *, wait_timeout: Optional[float] = 5.0) -> None:
        """
        Stop the processor gracefully:
        - Stop accepting new work.
        - Wake workers so they can exit when queue drained.
        - Wait for workers to finish (bounded by wait_timeout).
        """
        if not self._running.is_set():
            self.logger.debug("StreamProcessor not running.")
            return

        self.logger.info("Stopping StreamProcessor, waiting for workers to finish.")
        self._running.clear()

        # Put sentinels to wake workers if they are blocked on get.
        for _ in range(self.workers):
            try:
                # If queue is full, use put_nowait then fall back to await (rare).
                self.queue.put_nowait(_Sentinel)
            except asyncio.QueueFull:
                await self.queue.put(_Sentinel)

        # Wait for tasks to finish
        try:
            await asyncio.wait_for(asyncio.gather(*self._tasks, return_exceptions=True), timeout=wait_timeout)
        except asyncio.TimeoutError:
            self.logger.warning("Timeout while waiting for workers to finish; cancelling remaining tasks.")
            for t in self._tasks:
                t.cancel()
            # Allow cancellation to propagate
            await asyncio.gather(*self._tasks, return_exceptions=True)
        finally:
            self._tasks = []

        # Drain queue to avoid leaked items for future restarts
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except Exception:
                break

        self.logger.info("StreamProcessor stopped. processed=%d errors=%d", self.processed, self.errors)

    async def join(self, timeout: Optional[float] = None) -> None:
        """
        Wait until the queue is fully processed (queue.join()).
        Optional timeout in seconds.
        """
        waiter = asyncio.create_task(self.queue.join())
        try:
            if timeout is None:
                await waiter
            else:
                await asyncio.wait_for(waiter, timeout=timeout)
        finally:
            if not waiter.done():
                waiter.cancel()

    # Context manager support
    async def __aenter__(self) -> "StreamProcessor":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()

    # ---- Internal worker ----

    async def _worker_loop(self, worker_id: int) -> None:
        """
        Single worker loop: consumes the queue, routes signals through router.handle, and processes results.
        Exits when sentinel is received and no more queue items remain.
        """
        self.logger.debug("Worker[%d] started", worker_id)
        while True:
            try:
                signal = await self.queue.get()
            except asyncio.CancelledError:
                self.logger.debug("Worker[%d] cancelled", worker_id)
                break
            try:
                if signal is _Sentinel and not self._running.is_set():
                    # A sentinel to tell this worker to exit.
                    self.logger.debug("Worker[%d] received sentinel; exiting", worker_id)
                    self.queue.task_done()
                    break

                # Route the signal. Router.handle can be sync or async.
                try:
                    result = self.router.handle(signal)
                    if inspect.isawaitable(result):
                        result = await result  # type: ignore
                except Exception:
                    self.errors += 1
                    self.logger.exception("Error while handling signal: %r", signal)
                    # Continue to next item after marking task done
                    continue

                # Process/display the result (callback or default)
                try:
                    if self._result_handler:
                        maybe_awaitable = self._result_handler(result)
                        if inspect.isawaitable(maybe_awaitable):
                            await maybe_awaitable  # type: ignore
                    else:
                        self.display_result(result)
                    self.processed += 1
                except Exception:
                    self.errors += 1
                    self.logger.exception("Error while processing result: %r", result)
                finally:
                    # Mark queue task done regardless of result handler success
                    self.queue.task_done()

            except Exception:
                # Catch-all to keep worker alive on unexpected exceptions
                self.errors += 1
                self.logger.exception("Unexpected error in worker[%d]", worker_id)
                try:
                    self.queue.task_done()
                except Exception:
                    pass

        self.logger.debug("Worker[%d] finished", worker_id)

    # ---- Result formatting ----

    def display_result(self, result: Any) -> None:
        """
        Default result logging. Tries to format result safely without raising exceptions.
        Expecting result to be a mapping with keys like 'decision' and 'semantic' but will fall back to JSON.
        """
        try:
            if isinstance(result, dict):
                action = result.get("decision", {}).get("action", "<no-action>")
                semantic = result.get("semantic", {})
                intent = semantic.get("intent", "<no-intent>")
                confidence = semantic.get("confidence", None)
                if isinstance(confidence, (float, int)):
                    self.logger.info("[STREAM RESULT] %s (%s, confidence=%.2f)", action, intent, float(confidence))
                else:
                    self.logger.info("[STREAM RESULT] %s (%s)", action, intent)
            else:
                # Fallback: pretty-print
                self.logger.info("[STREAM RESULT] %s", json.dumps(result, default=str))
        except Exception:
            # Last-resort safe logging
            self.logger.exception("Failed to format stream result: %r", result)
            self.logger.info("[STREAM RESULT] %s", str(result))
