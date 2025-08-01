from strategy import RetaliationStrategy, SeverityBasedStrategy
from models import ThreatEvent
from logger import configure_logger

logger = configure_logger()

class RetaliationBot:
    def __init__(self, strategy: RetaliationStrategy = None):
        self.strategy = strategy or SeverityBasedStrategy()
        self._active = False

    def activate(self):
        self._active = True
        logger.info("Bot activated.")

    def deactivate(self):
        self._active = False
        logger.info("Bot deactivated.")

    def is_active(self):
        return self._active

    async def handle_event(self, event: ThreatEvent):
        if not self._active:
            logger.debug("Bot inactive. Event ignored.")
            return
        logger.info(f"Handling threat: {event}")
        await self.strategy.respond(event)