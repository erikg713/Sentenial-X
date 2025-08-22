# cli/memory_adapter.py
import sqlite3, json
from pathlib import Path
from .config import AGENT_MEMORY_DB, AGENT_ID

def _conn():
    return sqlite3.connect(AGENT_MEMORY_DB)

def log_command(action, params=None, result=None, status="success", actor="cli"):
    with _conn() as con:
        cur = con.cursor()
        cur.execute(
            "INSERT INTO commands(agent_id,action,params_json,result_json,status,actor) VALUES (?,?,?,?,?,?)",
            (AGENT_ID, action, json.dumps(params or {}), json.dumps(result or {}), status, actor)
        )
        return cur.lastrowid

def log_alert(type_, severity, details=None, related_command_id=None):
    with _conn() as con:
        cur = con.cursor()
        cur.execute(
            "INSERT INTO alerts(agent_id,type,severity,details_json,related_command_id) VALUES (?,?,?,?,?)",
            (AGENT_ID, type_, severity, json.dumps(details or {}), related_command_id)
        )
        return cur.lastrowid

def log_event(type_, source=None, risk=None, payload=None):
    with _conn() as con:
        cur = con.cursor()
        cur.execute(
            "INSERT INTO events(agent_id,type,source,risk,payload_json) VALUES (?,?,?,?,?)",
            (AGENT_ID, type_, source, risk, json.dumps(payload or {}))
        )
        return cur.lastrowid
