import os
import json
from datetime import datetime

LOG_PATH = "logs/agent_heartbeat.log"

def send_heartbeat(agent_id):
    os.makedirs("logs", exist_ok=True)
    heartbeat = {
        "agent": agent_id,
        "status": "online",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    with open(LOG_PATH, "a") as log:
        log.write(json.dumps(heartbeat) + "\n")
    print(f"[HEARTBEAT] {heartbeat['timestamp']}")
