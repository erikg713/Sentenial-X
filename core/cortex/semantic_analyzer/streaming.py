"""
core/semantic_analyzer/streaming.py

Real-time semantic analyzer streaming engine.
- Handles continuous ingestion of telemetry/security events.
- Supports pluggable analyzers (threat scoring, stealth detection, anomaly detection).
- Provides async queue-based processing.
- Broadcasts results to subscribers (e.g., WebSocket dashboard).
"""

import asyncio
import json
from typing import Dict, Any, Callable, List, Coroutine

from .scoring import score_evasion
from .anomaly import detect_anomaly


class StreamingEngine:
    """Async engine for handling streaming telemetry and semantic analysis."""

    def __init__(self, queue_size: int = 1000):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self.subscribers: List[Callable[[Dict[str, Any]], Coroutine]] = []
        self.analyzers: List[Callable[[Dict[str, Any]], Dict[str, Any]]] = [
            self._apply_scoring,
            self._apply_anomaly,
        ]
        self.running = False

    async def ingest(self, event: Dict[str, Any]):
        """Push an event into the queue."""
        await self.queue.put(event)

    async def _apply_scoring(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Apply stealth scoring."""
        event["evasion_score"] = score_evasion(event)
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
