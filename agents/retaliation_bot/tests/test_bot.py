import pytest
import asyncio
from bot import RetaliationBot
from models import ThreatEvent

@pytest.mark.asyncio
async def test_bot_handles_threat():
    bot = RetaliationBot()
    bot.activate()
    event = ThreatEvent("1.2.3.4", "Port Scan", 5)
    await bot.handle_event(event)
    assert bot.is_active()