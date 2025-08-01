import asyncio
import logging
import random
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

# Configure logger
logger = logging.getLogger("RetaliationBot")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# === Core Domain Classes ===

class ThreatEvent:
    """Represents a detected cyber threat."""
    def __init__(self, source_ip: str, vector: str, severity: int, details: Optional[Dict[str, Any]] = None):
        self.source_ip = source_ip
        self.vector = vector
        self.severity = severity
        self.timestamp = datetime.utcnow()
        self.details = details or {}

    def __str__(self):
        return (f"[{self.timestamp}] Threat from {self.source_ip} - "
                f"Vector: {self.vector}, Severity: {self.severity}, Details: {self.details}")


# === Action System ===

class RetaliationAction:
    """Abstract retaliation action."""
    async def execute(self, event: ThreatEvent):
        raise NotImplementedError("Subclasses must implement the execute() method.")


class BlacklistIPAction(RetaliationAction):
    async def execute(self, event: ThreatEvent):
        logger.warning(f"Blacklisting IP: {event.source_ip} [Vector: {event.vector}]")


class HoneypotRedirectAction(RetaliationAction):
    async def execute(self, event: ThreatEvent):
        logger.info(f"Redirecting {event.source_ip} to honeypot for further intelligence.")


class CounterScanAction(RetaliationAction):
    async def execute(self, event: ThreatEvent):
        logger.info(f"Initiating counter-scan on {event.source_ip}.")


class AlertAdminAction(RetaliationAction):
    async def execute(self, event: ThreatEvent):
        logger.info(f"Alerting administrator: {event}")


class MultiAction(RetaliationAction):
    """Executes multiple retaliation actions in parallel."""
    def __init__(self, actions: List[RetaliationAction]):
        self.actions = actions

    async def execute(self, event: ThreatEvent):
        await asyncio.gather(*(action.execute(event) for action in self.actions))


# === Strategy System ===

class RetaliationStrategy:
    """Abstract retaliation strategy."""
    async def respond(self, event: ThreatEvent):
        raise NotImplementedError("Subclasses must implement the respond() method.")


class SeverityBasedStrategy(RetaliationStrategy):
    """
    Retaliation strategy based on threat severity level.
    Custom thresholds and action lists can be supplied.
    """
    def __init__(self, thresholds: Optional[Dict[str, Tuple[int, int, List[RetaliationAction]]]] = None):
        self.thresholds = thresholds or {
            "low":    (0, 3,   [AlertAdminAction()]),
            "medium": (3, 7,   [AlertAdminAction(), BlacklistIPAction()]),
            "high":   (7, 11,  [AlertAdminAction(), BlacklistIPAction(), HoneypotRedirectAction(), CounterScanAction()])
        }

    async def respond(self, event: ThreatEvent):
        for label, (min_s, max_s, actions) in self.thresholds.items():
            if min_s <= event.severity < max_s:
                logger.debug(f"Severity level '{label}' matched for event: {event}")
                await asyncio.gather(*(action.execute(event) for action in actions))
                break


# === Retaliation Bot ===

class RetaliationBot:
    """Core engine that processes threats and executes strategies."""
    def __init__(self, strategy: Optional[RetaliationStrategy] = None):
        self.strategy = strategy or SeverityBasedStrategy()
        self._active = False

    def activate(self):
        self._active = True
        logger.info("RetaliationBot activated.")

    def deactivate(self):
        self._active = False
        logger.info("RetaliationBot deactivated.")

    def is_active(self) -> bool:
        return self._active

    def set_strategy(self, strategy: RetaliationStrategy):
        self.strategy = strategy
        logger.info(f"Strategy updated: {strategy.__class__.__name__}")

    async def handle_event(self, event: ThreatEvent):
        if not self._active:
            logger.debug("Bot is inactive. Event ignored.")
            return
        logger.info(f"Processing threat: {event}")
        await self.strategy.respond(event)


# === Simulation ===

async def simulate_threats(bot: RetaliationBot, num_events: int = 5):
    """Simulates random threat events."""
    vectors = ["Port Scan", "SQL Injection", "Brute Force", "Zero-Day Exploit"]
    for _ in range(num_events):
        event = ThreatEvent(
            source_ip=f"192.168.1.{random.randint(2, 254)}",
            vector=random.choice(vectors),
            severity=random.randint(1, 10),
            details={"user_agent": "MaliciousBot/1.0"}
        )
        await bot.handle_event(event)
        await asyncio.sleep(random.uniform(0.5, 2.0))


# === Entry Point ===

if __name__ == "__main__":
    bot = RetaliationBot()
    bot.activate()
    try:
        asyncio.run(simulate_threats(bot))
    finally:
        bot.deactivate()