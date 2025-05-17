import logging
import os
# utils/logger.py

import os
from datetime import datetime

LOG_FILE = "logs/threats.log"

def log_threat(timestamp, ip, threat_type, severity):
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        entry = f"[{timestamp}] {ip} | {threat_type} | {severity}\n"
        f.write(entry)
        
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

log_path = os.path.join(log_dir, "sentenialx.log")

logging.basicConfig(
    filename=log_path,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='a'
)

logger = logging.getLogger("SentenialX")
