#!/usr/bin/env python3
import sqlite3, os, json
from pathlib import Path

# ---- Mirror your config.py ----
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = DATA_DIR / "logs"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
REPORTS_DIR = DATA_DIR / "reports"
DB_PATH = DATA_DIR / "memory.db"

def ensure_dirs():
    for p in [LOGS_DIR, EMBEDDINGS_DIR, REPORTS_DIR]:
        p.mkdir(parents=True, exist_ok=True)

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.executescript("""
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS commands (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  agent_id TEXT NOT NULL,
  action TEXT NOT NULL,
  params_json TEXT,
  result_json TEXT,
  status TEXT DEFAULT 'success',
  actor TEXT DEFAULT 'cli',
  timestamp DATETIME DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_commands_ts ON commands(timestamp);
CREATE INDEX IF NOT EXISTS idx_commands_action ON commands(action);

CREATE TABLE IF NOT EXISTS alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  agent_id TEXT NOT NULL,
  type TEXT NOT NULL,
  severity TEXT NOT NULL,
  details_json TEXT,
  related_command_id INTEGER,
  timestamp DATETIME DEFAULT (datetime('now')),
  FOREIGN KEY(related_command_id) REFERENCES commands(id)
);

CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);

CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  agent_id TEXT NOT NULL,
  type TEXT NOT NULL,            -- e.g., 'telemetry','blindspot','cortex','wormgpt'
  source TEXT,                   -- e.g., 'network_monitor','/var/log/syslog'
  risk TEXT,                     -- e.g., 'low','medium','high'
  payload_json TEXT,
  timestamp DATETIME DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);

-- Optional: a table to persist streamed telemetry frames
CREATE TABLE IF NOT EXISTS telemetry_frames (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  agent_id TEXT NOT NULL,
  source TEXT NOT NULL,
  level TEXT,                    -- e.g., 'info','warning','high_severity'
  frame_json TEXT,
  timestamp DATETIME DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_frames_src_ts ON telemetry_frames(source, timestamp);

-- Helpful views
CREATE VIEW IF NOT EXISTS v_recent_alerts AS
SELECT id, type, severity, timestamp, json_extract(details_json,'$.entities') AS entities
FROM alerts ORDER BY timestamp DESC;

CREATE VIEW IF NOT EXISTS v_recent_commands AS
SELECT id, action, status, timestamp, json_extract(params_json,'$') AS params
FROM commands ORDER BY timestamp DESC;
""")
    con.commit()
    con.close()

def smoke_test():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    # insert one command + alert + event to verify writes
    cur.execute("INSERT INTO commands(agent_id, action, params_json) VALUES (?, ?, ?)",
                ("agent-001","bootstrap_check", json.dumps({"ok": True})))
    cmd_id = cur.lastrowid
    cur.execute("INSERT INTO alerts(agent_id, type, severity, details_json, related_command_id) VALUES (?,?,?,?,?)",
                ("agent-001","bootstrap_alert","low", json.dumps({"note":"init ok"}), cmd_id))
    cur.execute("INSERT INTO events(agent_id, type, source, risk, payload_json) VALUES (?,?,?,?,?)",
                ("agent-001","telemetry","bootstrap","low", json.dumps({"ping":"ok"})))
    con.commit()
    con.close()

if __name__ == "__main__":
    ensure_dirs()
    init_db()
    smoke_test()
    print(f"Initialized DB at: {DB_PATH}")
    print("Tables & sample rows inserted.")
