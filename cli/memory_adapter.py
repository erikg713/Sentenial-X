"""
cli/memory_adapter.py

Adapter layer for Sentenial-X memory operations.

Purpose:
- Provides a clean interface for CLI modules to read/write to memory.
- Supports multiple backends (currently SQLite via memory.py, future pluggable backends possible).
- Ensures consistent logging and error handling.
"""

import asyncio
from typing import Any, Dict, Optional, List
from cli import memory
from cli.logger import default_logger


# ------------------------------
# Adapter Class
# ------------------------------
class MemoryAdapter:
    """
    Async adapter for memory operations.
    Wraps memory.py for CLI modules.
    """

    def __init__(self):
        self.initialized = False

    async def init(self):
        if not self.initialized:
            await memory.init_db()
            self.initialized = True
            default_logger.info("MemoryAdapter initialized.")

    # --------------------------
    # Write Operations
    # --------------------------
    async def log_command(self, action: str, params: Optional[Dict[str, Any]] = None, result: Optional[Dict[str, Any]] = None):
        """Write a command entry."""
        await self.init()
        try:
            await memory.write_command(action, params, result)
        except Exception as e:
            default_logger.error(f"Failed to log command '{action}': {e}")

    async def log_alert(self, alert_type: str, severity: str, details: Optional[Dict[str, Any]] = None):
        """Write an alert entry."""
        await self.init()
        try:
            await memory.write_alert(alert_type, severity, details)
        except Exception as e:
            default_logger.error(f"Failed to log alert '{alert_type}': {e}")

    async def log_event(self, event_type: str, source: Optional[str] = None, details: Optional[Dict[str, Any]] = None, risk_level: Optional[str] = "low"):
        """Write a generic event entry."""
        await self.init()
        try:
            await memory.write_event(event_type, source, details, risk_level)
        except Exception as e:
            default_logger.error(f"Failed to log event '{event_type}': {e}")

    # --------------------------
    # Read / Query Operations
    # --------------------------
    async def get_recent_commands(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch recent commands."""
        await self.init()
        try:
            rows = await memory.fetch_recent_commands(limit)
            return [dict(row) if isinstance(row, dict) else row for row in rows]
        except Exception as e:
            default_logger.error(f"Failed to fetch recent commands: {e}")
            return []

    async def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch recent alerts."""
        await self.init()
        try:
            rows = await memory.fetch_recent_alerts(limit)
            return [dict(row) if isinstance(row, dict) else row for row in rows]
        except Exception as e:
            default_logger.error(f"Failed to fetch recent alerts: {e}")
            return []

    async def get_recent_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch recent events."""
        await self.init()
        try:
            rows = await memory.fetch_recent_events(limit)
            return [dict(row) if isinstance(row, dict) else row for row in rows]
        except Exception as e:
            default_logger.error(f"Failed to fetch recent events: {e}")
            return []


# ------------------------------
# Singleton Adapter Instance
# ------------------------------
_adapter_instance: Optional[MemoryAdapter] = None


def get_adapter() -> MemoryAdapter:
    """Return a singleton adapter instance."""
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = MemoryAdapter()
    return _adapter_instance


# ------------------------------
# Quick CLI Test
# ------------------------------
if __name__ == "__main__":
    import asyncio

    async def test_adapter():
        adapter = get_adapter()
        await adapter.log_command("test_cmd", {"foo": "bar"}, {"result": 123})
        await adapter.log_alert("test_alert", "high", {"reason": "unit test"})
        await adapter.log_event("test_event", "unit_test", {"details": "adapter test"}, "medium")

        cmds = await adapter.get_recent_commands()
        alerts = await adapter.get_recent_alerts()
        events = await adapter.get_recent_events()

        print("Recent commands:", cmds)
        print("Recent alerts:", alerts)
        print("Recent events:", events)

    asyncio.run(test_adapter())
