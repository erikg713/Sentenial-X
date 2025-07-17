import sqlite3
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger("SentenialX.Logger")
DB_PATH = Path("secure_db/data.db")

def log_threat_event(threat_type: str, source: str, payload: str, confidence: float):
    # Ensure parent directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS threat_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            threat_type TEXT,
            source TEXT,
            raw_payload TEXT,
            confidence REAL,
            mitigated INTEGER DEFAULT 0
        )
    """)

    # Insert threat data
    cursor.execute("""
        INSERT INTO threat_events (timestamp, threat_type, source, raw_payload, confidence)
        VALUES (?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        threat_type,
        source,
        payload,
        confidence
    ))

    # Commit and close
    conn.commit()
    conn.close()

    logger.info(f"Logged threat event: {threat_type} from {source}")
