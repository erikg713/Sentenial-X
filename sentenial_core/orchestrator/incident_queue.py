# sentenial_core/orchestrator/incident_queue.py

import asyncio
import heapq
import time
from typing import Any, Dict, Optional
from loguru import logger


class Incident:
    """
    Incident data structure for queueing.
    """
    def __init__(self, incident_id: str, severity: int, details: Dict[str, Any], timestamp: Optional[float] = None):
        self.incident_id = incident_id
        self.severity = severity  # Higher number means higher priority
        self.details = details
        self.timestamp = timestamp or time.time()

    def __lt__(self, other: "Incident"):
        # Priority queue sorting: by severity descending, then timestamp ascending
        if self.severity == other.severity:
            return self.timestamp < other.timestamp
        return self.severity > other.severity

    def __eq__(self, other: Any):
        if not isinstance(other, Incident):
            return False
        return self.incident_id == other.incident_id

    def __hash__(self):
        return hash(self.incident_id)

    def to_dict(self):
        return {
            "incident_id": self.incident_id,
            "severity": self.severity,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class IncidentQueue:
    """
    Async priority queue for managing incidents.
    Deduplicates by incident_id.
    """

    def __init__(self):
        self._queue = []
        self._incident_ids = set()
        self._cv = asyncio.Condition()

    async def enqueue(self, incident: Incident):
        async with self._cv:
            if incident.incident_id in self._incident_ids:
                logger.debug(f"Duplicate incident ignored: {incident.incident_id}")
                return
            heapq.heappush(self._queue, incident)
            self._incident_ids.add(incident.incident_id)
            logger.info(f"Enqueued incident: {incident.incident_id} severity={incident.severity}")
            self._cv.notify()

    async def dequeue(self) -> Optional[Incident]:
        async with self._cv:
            while not self._queue:
                await self._cv.wait()
            incident = heapq.heappop(self._queue)
            self._incident_ids.remove(incident.incident_id)
            logger.info(f"Dequeued incident: {incident.incident_id}")
            return incident

    async def peek(self) -> Optional[Incident]:
        async with self._cv:
            if not self._queue:
                return None
            return self._queue[0]

    async def size(self) -> int:
        async with self._cv:
            return len(self._queue)

    async def clear(self):
        async with self._cv:
            self._queue.clear()
            self._incident_ids.clear()
            logger.info("Cleared incident queue")
            self._cv.notify_all()

    # Optional: Persistence hooks (implement as needed)
    async def save_to_storage(self):
        # Serialize queue to persistent storage (DB/file)
        pass

    async def load_from_storage(self):
        # Load queue state from persistent storage
        pass
