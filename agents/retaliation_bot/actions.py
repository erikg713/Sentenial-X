from abc import ABC, abstractmethod
from models import ThreatEvent
from logger import configure_logger

logger = configure_logger()

class RetaliationAction(ABC):
    @abstractmethod
    async def execute(self, event: ThreatEvent):
        pass

class BlacklistIPAction(RetaliationAction):
    async def execute(self, event: ThreatEvent):
        logger.warning(f"Blacklisting IP: {event.source_ip}")

class HoneypotRedirectAction(RetaliationAction):
    async def execute(self, event: ThreatEvent):
        logger.info(f"Redirecting {event.source_ip} to honeypot")

class CounterScanAction(RetaliationAction):
    async def execute(self, event: ThreatEvent):
        logger.info(f"Counter-scanning {event.source_ip}")

class AlertAdminAction(RetaliationAction):
    async def execute(self, event: ThreatEvent):
        logger.info(f"Alerting admin: {event}")