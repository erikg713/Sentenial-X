# cli/memory.py
import aiosqlite
import asyncio
from pathlib import Path
from .config import AGENT_MEMORY_DB

DB_PATH = AGENT_MEMORY_DB

async def init_db():
    """
    Initialize the SQLite memory DB if not exists.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS agent_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT,
                action TEXT,
                meta TEXT,
                timestamp TEXT
            )
        """)
        await db.commit()

async def enqueue_command(agent_id: str, action: str, meta: dict = None):
    """
    Log a command/action into agent memory.
    """
    meta_json = "{}" if meta is None else json.dumps(meta)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO agent_memory (agent_id, action, meta, timestamp)
            VALUES (?, ?, ?, datetime('now'))
        """, (agent_id, action, meta_json))
        await db.commit()

async def fetch_memory(agent_id: str = None, limit: int = 50):
    """
    Fetch recent memory logs.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        if agent_id:
            cursor = await db.execute("""
                SELECT * FROM agent_memory
                WHERE agent_id = ?
                ORDER BY id DESC
                LIMIT ?
            """, (agent_id, limit))
        else:
            cursor = await db.execute("""
                SELECT * FROM agent_memory
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
        rows = await cursor.fetchall()
    return rows

# Synchronous helper for CLI scripts
def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)
