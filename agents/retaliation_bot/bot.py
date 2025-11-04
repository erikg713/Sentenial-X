import asyncio
from typing import Optional
from strategy import RetaliationStrategy, SeverityBasedStrategy
from models import ThreatEvent
from logger import configure_logger

logger = configure_logger()

class RetaliationBot:
    """
    A bot class for handling threat events using a specified retaliation strategy.
    It can be activated/deactivated and processes events asynchronously.
    """

    def __init__(self, strategy: Optional[RetaliationStrategy] = None):
        """
        Initialize the RetaliationBot.

        :param strategy: Optional retaliation strategy. Defaults to SeverityBasedStrategy if not provided.
        :raises ValueError: If the provided strategy is not an instance of RetaliationStrategy.
        """
        if strategy is None:
            self._strategy = SeverityBasedStrategy()
        elif isinstance(strategy, RetaliationStrategy):
            self._strategy = strategy
        else:
            raise ValueError("Strategy must be an instance of RetaliationStrategy or its subclass.")
        self._active = False

    @property
    def strategy(self) -> RetaliationStrategy:
        """
        Get the current retaliation strategy.
        """
        return self._strategy

    @strategy.setter
    def strategy(self, new_strategy: RetaliationStrategy):
        """
        Set a new retaliation strategy.

        :param new_strategy: The new strategy to use.
        :raises ValueError: If the new strategy is not an instance of RetaliationStrategy.
        """
        if not isinstance(new_strategy, RetaliationStrategy):
            raise ValueError("New strategy must be an instance of RetaliationStrategy or its subclass.")
        self._strategy = new_strategy
        logger.info("Strategy updated.")

    @property
    def active(self) -> bool:
        """
        Check if the bot is active.
        """
        return self._active

    def activate(self):
        """
        Activate the bot.
        """
        if self._active:
            logger.warning("Bot is already active.")
            return
        self._active = True
        logger.info("Bot activated.")

    def deactivate(self):
        """
        Deactivate the bot.
        """
        if not self._active:
            logger.warning("Bot is already deactivated.")
            return
        self._active = False
        logger.info("Bot deactivated.")

    async def handle_event(self, event: ThreatEvent):
        """
        Handle a threat event asynchronously if the bot is active.

        :param event: The ThreatEvent to handle.
        :raises ValueError: If the event is not a ThreatEvent instance.
        """
        if not isinstance(event, ThreatEvent):
            raise ValueError("Event must be an instance of ThreatEvent.")
        if not self._active:
            logger.debug("Bot inactive. Event ignored.")
            return
        logger.info(f"Handling threat: {event}")
        try:
            await self._strategy.respond(event)
        except Exception as e:
            logger.error(f"Error responding to event {event}: {e}")
            # Optionally, re-raise or handle based on requirements
