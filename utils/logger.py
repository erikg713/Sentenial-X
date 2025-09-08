import logging
import os
from datetime import datetime
import structlog
import json

def log_metrics(metrics: dict, filename="training_log.json"):
    """Save training metrics to file."""
    metrics["timestamp"] = datetime.utcnow().isoformat()
    with open(filename, "a") as f:
        f.write(json.dumps(metrics) + "\n")
    print(f"[+] Metrics logged: {metrics}")

logger = structlog.get_logger()

logger.info("threat_detected", ip="192.168.0.3", intent="injection", severity="high")

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
"""
Sentenial-X A.I. Logging Utility

Provides a unified, robust logging interface for application and security event logging.
Handles log directory creation, log rotation, and separate threat logging.
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

# === Configuration ===
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
APP_LOG_PATH = os.path.join(LOG_DIR, "sentenialx.log")
THREAT_LOG_PATH = os.path.join(LOG_DIR, "threats.log")
MAX_LOG_SIZE = 2 * 1024 * 1024  # 2 MB
BACKUP_COUNT = 5

# === Ensure log directory exists (atomic, threadsafe) ===
os.makedirs(LOG_DIR, exist_ok=True)

# === Logger Factory ===
def get_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured logger with rotation.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Prevent duplicate handlers in interactive/debug environments
    if not any(isinstance(h, RotatingFileHandler) and h.baseFilename == os.path.abspath(log_file) for h in logger.handlers):
        handler = RotatingFileHandler(
            log_file, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT, encoding="utf-8"
        )
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger

# Main application logger (DEBUG for full traceability)
app_logger = get_logger("SentenialX", APP_LOG_PATH, level=logging.DEBUG)

def log_event(level: str, msg: str):
    """
    Log a general event to the application logger.
    """
    level = level.lower()
    log_func = getattr(app_logger, level, app_logger.info)
    log_func(msg)

def log_threat(ip: str, threat_type: str, severity: str, timestamp: str = None):
    """
    Log a threat event to a dedicated threats log and to main logger.
    """
    if not timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {ip} | {threat_type} | {severity}\n"
    with open(THREAT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(entry)
    app_logger.warning(f"Threat event: {ip} | {threat_type} | {severity}")

# === Example Usage ===
if __name__ == "__main__":
    log_event("info", "Sentenial-X logger initialized.")
    log_threat("10.0.0.7", "Ransomware Detected", "CRITICAL")
    log_event("debug", "Debugging log output works.")
