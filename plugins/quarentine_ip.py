# plugins/quarantine_ip.py

from sentenial_core.adapters.db_adapter import DBAdapter
from datetime import datetime

def register(register_command):
    register_command("quarantine_ip", quarantine_ip)

async_db = DBAdapter()

def quarantine_ip(ip: str) -> str:
    """
    Mark an IP as quarantined in the agent’s memory store.
    """
    if not ip:
        return "Usage: quarantine_ip <IP_ADDRESS>"

    event = {
        "action": "quarantine",
        "target": ip,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    # Note: this runs sync—fire-and-forget the async call
    import asyncio
    asyncio.get_event_loop().create_task(async_db.log_memory(event))

    return f"IP {ip} has been quarantined."
