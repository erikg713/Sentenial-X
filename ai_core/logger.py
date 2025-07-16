# sentenialx/ai_core/logger.py

import sqlite3
from datetime import datetime

DB_PATH = "secure_db/data.db"

def log_threat_event(threat_type, source, payload, confidence):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS threat_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        threat_type TEXT,
        source TEXT,
        raw_payload TEXT,
        confidence REAL,
        mitigated INTEGER DEFAULT 0
    )""")
    cursor.execute("INSERT INTO threat_events (timestamp, threat_type, source, raw_payload, confidence) VALUES (?, ?, ?, ?, ?)",
                   (datetime.utcnow().isoformat(), threat_type, source, payload, confidence))
    conn.commit()
    conn.close()
