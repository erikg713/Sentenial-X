import sqlite3
from pathlib import Path

DB_PATH = Path("secure_db/data.db")

def get_recent_threats(limit=50):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM threat_events ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows