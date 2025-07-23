import aiosqlite
import json
from datetime import datetime
from typing import List, Tuple, Any
from config import DB_PATH

class DBAdapter:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    async def init_db(self) -> None:
        ddl = [
            """
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                event TEXT NOT NULL
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS commands (
                id INTEGER PRIMARY KEY,
                agent_id TEXT NOT NULL,
                command_text TEXT NOT NULL,
                processed INTEGER NOT NULL DEFAULT 0
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS heartbeat (
                id INTEGER PRIMARY KEY,
                agent TEXT NOT NULL,
                status TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
            """
        ]
        async with aiosqlite.connect(self.db_path) as db:
            for stmt in ddl:
                await db.execute(stmt)
            await db.commit()

    async def log_memory(self, event: dict[str, Any]) -> None:
        payload = json.dumps(event)
        ts = datetime.utcnow().isoformat() + 'Z'
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO memory (timestamp, event) VALUES (?, ?)",
                (ts, payload)
            )
            await db.commit()

    async def fetch_memory(self) -> List[dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT event FROM memory")
            rows = await cursor.fetchall()
        return [json.loads(row[0]) for row in rows]

    async def enqueue_command(self, agent_id: str, command: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO commands (agent_id, command_text) VALUES (?, ?)",
                (agent_id, command)
            )
            await db.commit()

    async def fetch_commands(self, agent_id: str) -> List[Tuple[int, str]]:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT id, command_text FROM commands WHERE agent_id = ? AND processed = 0",
                (agent_id,)
            )
            return await cursor.fetchall()

    async def mark_command_processed(self, cmd_id: int) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE commands SET processed = 1 WHERE id = ?",
                (cmd_id,)
            )
            await db.commit()

    async def log_heartbeat(self, agent: str, status: str, timestamp: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO heartbeat (agent, status, timestamp) VALUES (?, ?, ?)",
                (agent, status, timestamp)
            )
            await db.commit()
