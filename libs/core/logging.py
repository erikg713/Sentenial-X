# libs/core/logging.py

import logging
import os
import json
from logging.handlers import RotatingFileHandler

try:
    from colorlog import ColoredFormatter
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False

DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_FILE = "sentenialx.log"
MAX_BYTES = 5 * 1024 * 1024  # 5MB
BACKUP_COUNT = 3

MODULE_LOG_LEVELS = {}
CONFIG_PATH = os.getenv("LOG_CONFIG_FILE", "config/logging_levels.json")

if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r") as f:
            MODULE_LOG_LEVELS = json.load(f)
    except Exception as e:
        print(f"[LOGGING] Failed to load log config: {e}")

def setup_logger(name: str = "sentenialx", default_level: str = "INFO") -> logging.Logger:
    os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
    logger = logging.getLogger(name)

    module_base = name.split(".")[0]
    level = MODULE_LOG_LEVELS.get(module_base, default_level).upper()
    logger.setLevel(getattr(logging, level, logging.INFO))

    if not any(isinstance(h, (logging.StreamHandler, RotatingFileHandler)) for h in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(_get_colored_formatter() if COLORLOG_AVAILABLE else _get_plain_formatter())
        logger.addHandler(console_handler)

        file_handler = RotatingFileHandler(
            os.path.join(DEFAULT_LOG_DIR, DEFAULT_LOG_FILE),
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT
        )
        file_handler.setFormatter(_get_plain_formatter())
        logger.addHandler(file_handler)

    return logger

def _get_plain_formatter() -> logging.Formatter:
    return logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] :: %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

def _get_colored_formatter() -> ColoredFormatter:
    return ColoredFormatter(
        fmt="%(log_color)s[%(asctime)s] [%(levelname)s] [%(name)s] :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red"
        }
    )

if __name__ == "__main__":
    os.environ["LOG_CONFIG_FILE"] = "config/logging_levels.json"
    setup_logger("core.analyzer").debug("Core module debug")
    setup_logger("api.gateway").info("API module info")
    setup_logger("engine.detector").warning("Engine warning")
    setup_logger("telemetry.receiver").error("Telemetry error")
