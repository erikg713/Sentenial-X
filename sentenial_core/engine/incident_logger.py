# engine/incident_logger.py

import sqlite3
from datetime import datetime

class IncidentLogger:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self._setup()

    def _setup(self):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT,
            message TEXT,
            timestamp TEXT
        )''')
        self.conn.commit()

    def log(self, category, message):
        c = self.conn.cursor()
        timestamp = datetime.utcnow().isoformat()
        c.execute("INSERT INTO incidents (category, message, timestamp) VALUES (?, ?, ?)",
                  (category, message, timestamp))
        self.conn.commit()