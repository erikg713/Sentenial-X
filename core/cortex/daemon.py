# sentenial_x/core/cortex/daemon.py

import asyncio
import signal
import logging
from .stream_handler import StreamHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cortex.daemon")

class CortexDaemon:
    def __init__(self, mode="kafka", kafka_topic="pinet_logs", kafka_bootstrap="localhost:9092", ws_url=None):
        self.mode = mode
        self.kafka_topic = kafka_topic
        self.kafka_bootstrap = kafka_bootstrap
        self.ws_url = ws_url
        self.stream_handler = StreamHandler(mode=mode, kafka_topic=kafka_topic, kafka_bootstrap=kafka_bootstrap, ws_url=ws_url)
        self.loop = asyncio.get_event_loop()
        self.should_exit = False

    def run(self):
        def shutdown_handler():
            logger.info("Shutdown signal received, stopping daemon...")
            self.should_exit = True

        # Register signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            self.loop.add_signal_handler(sig, shutdown_handler)

        logger.info(f"Starting CortexDaemon with mode={self.mode}")
        if self.mode == "kafka":
            self._run_kafka()
        elif self.mode == "websocket":
            self.loop.run_until_complete(self._run_websocket())
        else:
            logger.error(f"Unknown mode: {self.mode}")

    def _run_kafka(self):
        for message in self.stream_handler._start_kafka_stream():
            if self.should_exit:
                break
            self.stream_handler.process_event(message.value)
        logger.info("CortexDaemon stopped.")

    async def _run_websocket(self):
        async def runner():
            await self.stream_handler._start_ws_stream()
        task = self.loop.create_task(runner())
        while not self.should_exit:
            await asyncio.sleep(1)
        task.cancel()
        logger.info("CortexDaemon stopped.")

if __name__ == "__main__":
    daemon = CortexDaemon()
    daemon.run()

