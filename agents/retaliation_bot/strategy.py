import asyncio
from typing import Dict, List, Optional, Tuple
from models import ThreatEvent
from actions import RetaliationAction, AlertAdminAction, BlacklistIPAction, HoneypotRedirectAction, CounterScanAction
from logger import configure_logger

logger = configure_logger()

class RetaliationStrategy:
    async def respond(self, event: ThreatEvent):
        raise NotImplementedError

class SeverityBasedStrategy(RetaliationStrategy):
    def __init__(self, thresholds: Optional[Dict[str, Tuple[int, int, List[RetaliationAction]]]] = None):
        self.thresholds = thresholds or {
            "low": (0, 3, [AlertAdminAction()]),
            "medium": (3, 7, [AlertAdminAction(), BlacklistIPAction()]),
            "high": (7, 11, [AlertAdminAction(), BlacklistIPAction(), HoneypotRedirectAction(), CounterScanAction()])
        }

    async def respond(self, event: ThreatEvent):
        for label, (min_s, max_s, actions) in self.thresholds.items():
            if min_s <= event.severity < max_s:
                logger.debug(f"Severity '{label}' matched.")
                await asyncio.gather(*(action.execute(event) for action in actions))
                break