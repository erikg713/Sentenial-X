import asyncio
import logging

class StreamProcessor:
    def __init__(self, router):
        self.router = router
        self.queue = asyncio.Queue()
        self.running = False

    async def add_signal(self, signal):
        await self.queue.put(signal)

    async def start_stream(self):
        self.running = True
        while self.running:
            signal = await self.queue.get()
            result = self.router.handle(signal)
            self.display_result(result)

    def display_result(self, result):
        logging.info(f"[STREAM RESULT] {result['decision']['action']} ({result['semantic']['intent']}, confidence={result['semantic']['confidence']:.2f})")

    def stop_stream(self):
        self.running = False

