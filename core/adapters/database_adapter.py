"""
core/adapters/database_adapter.py

Sentenial-X Adapters Database Adapter Module - provides persistent storage adapter using SQLite
(or configurable DB) for logging events, ledgers, and forensic data. Supports asynchronous operations,
immutability via hash checks, and integration with modules like LedgerSequencer and IncidentQueue.
Handles schema creation, inserts, queries, and verification for tamper-evidence.
"""

import asyncio
import aiosqlite
import time
import json
import hashlib
from typing import Dict, List, Any, Optional
from cli.logger import default_logger

# Database schema for events/ledgers (expandable)
DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    event_id TEXT UNIQUE NOT NULL,
    sequence_id INTEGER,
    data JSON NOT NULL,
    hash TEXT NOT NULL,
    prev_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sequence ON events (sequence_id);
CREATE INDEX IF NOT EXISTS idx_timestamp ON events (timestamp);
"""

class DatabaseAdapter:
    """
    Asynchronous database adapter for persistent storage of forensic and incident data.
    Uses SQLite via aiosqlite for async ops; configurable for other DBs (e.g., PostgreSQL).
    
    :param db_path: Path to SQLite database file
    """
    def __init__(self, db_path: str = "sentenialx.db"):
        self.db_path = db_path
        self.logger = default_logger
        self.conn: Optional[aiosqlite.Connection] = None
        self._init_task = asyncio.create_task(self._initialize())

    async def _initialize(self):
        """Initialize DB connection and schema."""
        self.conn = await aiosqlite.connect(self.db_path)
        await self.conn.executescript(DB_SCHEMA)
        await self.conn.commit()
        self.logger.info(f"Database adapter initialized: {self.db_path}")

    async def _compute_hash(self, data: Dict[str, Any], prev_hash: str = "genesis") -> str:
        """Compute SHA-256 hash for data integrity."""
        data_str = json.dumps(data, sort_keys=True) + prev_hash
        return hashlib.sha256(data_str.encode()).hexdigest()

    async def log_command(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log an event/command to the database with sequencing and hashing.
        
        :param event_data: Event data to persist
        :return: Inserted event with ID
        """
        await self._init_task
        now = time.time()
        
        # Get last sequence and hash
        async with self.conn.execute("SELECT sequence_id, hash FROM events ORDER BY id DESC LIMIT 1") as cursor:
            last = await cursor.fetchone()
            sequence_id = last[0] + 1 if last else 1
            prev_hash = last[1] if last else "genesis"
        
        event_id = f"evt_{int(now)}_{sequence_id}"
        event = {
            "timestamp": now,
            "event_id": event_id,
            "sequence_id": sequence_id,
            "data": json.dumps(event_data),
            "prev_hash": prev_hash,
            "hash": await self._compute_hash(event_data, prev_hash)
        }
        
        await self.conn.execute(
            "INSERT INTO events (timestamp, event_id, sequence_id, data, hash, prev_hash) VALUES (?, ?, ?, ?, ?, ?)",
            (event["timestamp"], event["event_id"], event["sequence_id"], event["data"], event["hash"], event["prev_hash"])
        )
        await self.conn.commit()
        
        self.logger.info(f"Logged event {event_id} to database")
        return event

    async def query_events(self, start_seq: Optional[int] = None, end_seq: Optional[int] = None,
                           filter_key: Optional[str] = None, filter_value: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Query events by sequence range or JSON data filter.
        
        :param start_seq: Start sequence ID (inclusive)
        :param end_seq: End sequence ID (inclusive)
        :param filter_key: JSON key to filter (e.g., "action")
        :param filter_value: Value to match
        :return: List of matching events
        """
        await self._init_task
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        
        if start_seq is not None:
            query += " AND sequence_id >= ?"
            params.append(start_seq)
        if end_seq is not None:
            query += " AND sequence_id <= ?"
            params.append(end_seq)
        if filter_key and filter_value:
            query += f" AND json_extract(data, '$.{filter_key}') = ?"
            params.append(filter_value)
        
        query += " ORDER BY sequence_id ASC"
        
        async with self.conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            events = []
            for row in rows:
                event = {
                    "id": row[0],
                    "timestamp": row[1],
                    "event_id": row[2],
                    "sequence_id": row[3],
                    "data": json.loads(row[4]),
                    "hash": row[5],
                    "prev_hash": row[6]
                }
                events.append(event)
        
        self.logger.debug(f"Queried {len(events)} events from database")
        return events

    async def verify_chain_integrity(self) -> Dict[str, Any]:
        """
        Verify the integrity of the event chain by recomputing hashes.
        
        :return: Verification report
        """
        await self._init_task
        events = await self.query_events()
        issues = []
        prev_hash = "genesis"
        
        for event in events:
            recomputed = await self._compute_hash(event["data"], prev_hash)
            if recomputed != event["hash"]:
                issues.append(f"Hash mismatch at seq {event['sequence_id']}: expected {event['hash']}, got {recomputed}")
            prev_hash = event["hash"]
        
        valid = len(issues) == 0
        self.logger.info(f"Chain integrity: {'Valid' if valid else 'Issues found'} ({len(issues)})")
        return {"valid": valid, "issues": issues, "total_events": len(events)}

    async def close(self):
        """Close the database connection."""
        if self.conn:
            await self.conn.close()
        self.logger.info("Database adapter closed")

# Example usage (integrate with LedgerSequencer, etc.)
async def example_persistence():
    """Demo: Log and query events."""
    adapter = DatabaseAdapter()
    
    # Mock event
    event_data = {
        "action": "wormgpt_detection",
        "prompt": "Test prompt",
        "risk_level": "high"
    }
    
    logged = await adapter.log_command(event_data)
    print(json.dumps(logged, indent=2))
    
    # Query
    events = await adapter.query_events(start_seq=1)
    print(f"Queried {len(events)} events")
    
    # Verify
    integrity = await adapter.verify_chain_integrity()
    print(f"Integrity valid: {integrity['valid']}")
    
    await adapter.close()

if __name__ == "__main__":
    asyncio.run(example_persistence())
