"""
cli/memory.py

Async memory backend for Sentenial-X CLI.

Responsibilities:
- Persist CLI actions, alerts, telemetry, and events
- Supports SQLite by default, pluggable to other DB backends
- Fully async for production-grade performance
- Integrated with logger.py for structured logging
"""

import aiosqlite
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional
from cli.logger import default_logger

# ------------------------------
# Configuration
# ------------------------------
DB_PATH = os.getenv("SENTENIAL_DB", "sentenialx.db")


# ------------------------------
# Database Helper
# ------------------------------
async def get_connection():
    """Returns an async SQLite connection."""
    conn = await aiosqlite.connect(DB_PATH)
    await conn.execute("PRAGMA foreign_keys = ON;")
    return conn


# ------------------------------
# Schema Initialization
# ------------------------------
async def init_db():
    """Initialize tables if they don't exist."""
    default_logger.info("Initializing memory database...")
    async with await get_connection() as db:
        await db.execute("""
        CREATE TABLE IF NOT EXISTS commands (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT NOT NULL,
            params_json TEXT,
            result_json TEXT,
            timestamp TEXT NOT NULL
        );
        """)
        await db.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            severity TEXT NOT NULL,
            details TEXT,
            timestamp TEXT NOT NULL
        );
        """)
        await db.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            source TEXT,
            details TEXT,
            risk_level TEXT,
            timestamp TEXT NOT NULL
        );
        """)
        await db.commit()
    default_logger.info("Memory database initialized.")


# ------------------------------
# Async Memory Operations
# ------------------------------
async def write_command(action: str, params: Optional[Dict[str, Any]] = None, result: Optional[Dict[str, Any]] = None):
    """Store a CLI command execution."""
    timestamp = datetime.utcnow().isoformat()
    async with await get_connection() as db:
        await db.execute(
            "INSERT INTO commands (action, params_json, result_json, timestamp) VALUES (?, ?, ?, ?)",
            (action, json.dumps(params or {}), json.dumps(result or {}), timestamp)
        )
        await db.commit()
    default_logger.debug(f"Command logged: {action}")


async def write_alert(alert_type: str, severity: str, details: Optional[Dict[str, Any]] = None):
    """Store an alert."""
    timestamp = datetime.utcnow().isoformat()
    async with await get_connection() as db:
        await db.execute(
            "INSERT INTO alerts (type, severity, details, timestamp) VALUES (?, ?, ?, ?)",
            (alert_type, severity, json.dumps(details or {}), timestamp)
        )
        await db.commit()
    default_logger.debug(f"Alert logged: {alert_type} / {severity}")


async def write_event(event_type: str, source: Optional[str] = None, details: Optional[Dict[str, Any]] = None, risk_level: Optional[str] = "low"):
    """Store a generic event (e.g., telemetry or blind spot detection)."""
    timestamp = datetime.utcnow().isoformat()
    async with await get_connection() as db:
        await db.execute(
            "INSERT INTO events (event_type, source, details, risk_level, timestamp) VALUES (?, ?, ?, ?, ?)",
            (event_type, source or "unknown", json.dumps(details or {}), risk_level, timestamp)
        )
        await db.commit()
    default_logger.debug(f"Event logged: {event_type} / {risk_level}")


# ------------------------------
# Query Helpers
# ------------------------------
async def fetch_recent_commands(limit: int = 10):
    async with await get_connection() as db:
        cursor = await db.execute("SELECT * FROM commands ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = await cursor.fetchall()
    return rows


async def fetch_recent_alerts(limit: int = 10):
    async with await get_connection() as db:
        cursor = await db.execute("SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = await cursor.fetchall()
    return rows


async def fetch_recent_events(limit: int = 10):
    async with await get_connection() as db:
        cursor = await db.execute("SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = await cursor.fetchall()
    return rows


# ------------------------------
# CLI Test Run
# ------------------------------
if __name__ == "__main__":
    import asyncio

    async def test_memory():
        await init_db()
        await write_command("test_command", {"foo": "bar"}, {"result": 123})
        await write_alert("test_alert", "high", {"reason": "unit test"})
        await write_event("test_event", "unit_test", {"details": "testing"}, "medium")

        cmds = await fetch_recent_commands()
        alerts = await fetch_recent_alerts()
        events = await fetch_recent_events()

        print("Recent commands:", cmds)
        print("Recent alerts:", alerts)
        print("Recent events:", events)

    asyncio.run(test_memory())
