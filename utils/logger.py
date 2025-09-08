"""
Sentenial-X A.I. Logging Utility

Provides a unified, robust logging interface for application and security event logging.
Handles log directory creation, log rotation, and separate threat logging.
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
import threading
import structlog
import json

# === Configuration ===
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
APP_LOG_PATH = os.path.join(LOG_DIR, "sentenialx.log")
THREAT_LOG_PATH = os.path.join(LOG_DIR, "threats.log")
METRICS_LOG_PATH = os.path.join(LOG_DIR, "training_log.json")
MAX_LOG_SIZE = 2 * 1024 * 1024  # 2 MB
BACKUP_COUNT = 5

# === Ensure log directory exists (atomic, threadsafe) ===
_log_dir_lock = threading.Lock()
def ensure_log_dir():
    with _log_dir_lock:
        os.makedirs(LOG_DIR, exist_ok=True)
ensure_log_dir()

# === Logger Factory (with rotation and deduplication) ===
def get_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured logger with rotation (deduplicated).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    log_file = os.path.abspath(log_file)
    # Deduplicate handlers (avoid duplicate logs in debug/interactive mode)
    if not any(
        isinstance(h, RotatingFileHandler) and os.path.abspath(h.baseFilename) == log_file
        for h in logger.handlers
    ):
        handler = RotatingFileHandler(
            log_file, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT, encoding="utf-8"
        )
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger

# === Main application logger (DEBUG for full traceability) ===
app_logger = get_logger("SentenialX", APP_LOG_PATH, level=logging.DEBUG)

# === Structlog for structured event logging (optional, for integrations) ===
struct_logger = structlog.get_logger()

# === General Event Logging ===
def log_event(level: str, msg: str, **kwargs):
    """
    Log a general event to the application logger and optionally to structlog.
    """
    level = level.lower()
    log_func = getattr(app_logger, level, app_logger.info)
    log_func(msg)
    if kwargs:
        struct_logger.info(msg, **kwargs)

# === Threat Event Logging ===
def log_threat(ip: str, threat_type: str, severity: str, timestamp: str = None, extra: dict = None):
    """
    Log a threat event to a dedicated threats log and to the main logger.
    """
    if not timestamp:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {ip} | {threat_type} | {severity}\n"
    with open(THREAT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(entry)
    # Log to app_logger with more context
    app_logger.warning(f"Threat event: {ip} | {threat_type} | {severity}")
    # Structured logging for integrations/analytics
    event_data = {
        "ip": ip,
        "threat_type": threat_type,
        "severity": severity,
        "timestamp": timestamp,
    }
    if extra:
        event_data.update(extra)
    struct_logger.warning("threat_detected", **event_data)

# === Metrics Logging (training, etc.) ===
def log_metrics(metrics: dict, filename: str = METRICS_LOG_PATH):
    """
    Save training metrics to file (JSON lines).
    """
    ensure_log_dir()
    metrics = {**metrics, "timestamp": datetime.utcnow().isoformat()}
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics) + "\n")
    app_logger.info(f"Metrics logged: {metrics}")

# === Example Usage ===
if __name__ == "__main__":
    log_event("info", "Sentenial-X logger initialized.")
    log_threat("10.0.0.7", "Ransomware Detected", "CRITICAL")
    log_event("debug", "Debugging log output works.")
    log_metrics({"accuracy": 0.98, "loss": 0.04, "epoch": 17})
