import os
import json
import time
import random
import logging
from pathlib import Path
from datetime import datetime

# === Config ===
AGENT_ID = os.getenv("AGENT_ID", "sentenial-x-ai-bot")
BASE_DIR = Path(__file__).parent
LOG_PATH = BASE_DIR / "logs" / "agent_heartbeat.log"
COMMAND_QUEUE = BASE_DIR / "commands" / "agent_commands.json"
HEARTBEAT_INTERVAL = 10  # seconds

# === Logging Setup ===
os.makedirs(LOG_PATH.parent, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, mode='a')
    ]
)

def send_heartbeat():
    heartbeat = {
        "agent": AGENT_ID,
        "status": "online",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    logging.info(f"Heartbeat sent: {heartbeat['timestamp']}")

def listen_for_commands():
    if not COMMAND_QUEUE.exists():
        return
    try:
        with COMMAND_QUEUE.open("r") as f:
            commands = json.load(f)
        if AGENT_ID in commands:
            for cmd in commands[AGENT_ID]:
                logging.info(f"Executing command: {cmd}")
            # Clear commands for this agent
            del commands[AGENT_ID]
            with COMMAND_QUEUE.open("w") as f:
                json.dump(commands, f, indent=2)
    except Exception as e:
        logging.error(f"Command processing failed: {e}")

def run_agent():
    logging.info(f"Agent '{AGENT_ID}' booted.")
    while True:
        send_heartbeat()
        listen_for_commands()
        time.sleep(HEARTBEAT_INTERVAL)

if __name__ == "__main__":
    run_agent()