import asyncio
from loguru import logger
from sentenial_core.orchestrator.incident_queue import IncidentQueue, Incident


class IncidentProcessor:
    def __init__(self, queue: IncidentQueue):
        self.queue = queue
        self._running = False

    async def start(self):
        self._running = True
        logger.info("Incident processor started")
        while self._running:
            incident = await self.queue.dequeue()
            await self.handle_incident(incident)

    async def handle_incident(self, incident: Incident):
        logger.info(f"Processing incident {incident.incident_id} with severity {incident.severity}")
        # TODO: Add actual incident handling logic (alerts, notifications, mitigation)
        await asyncio.sleep(1)  # simulate work

    async def stop(self):
        self._running = False
        logger.info("Incident processor stopped")


async def main():
    queue = IncidentQueue()
    processor = IncidentProcessor(queue)

    # Start processor in background
    processor_task = asyncio.create_task(processor.start())

    # Enqueue some incidents
    await queue.enqueue(Incident("INC001", 10, {"desc": "Ransomware detected"}))
    await queue.enqueue(Incident("INC002", 5, {"desc": "Phishing attempt"}))

    await asyncio.sleep(5)  # Let processor work

    await processor.stop()
    await processor_task


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
