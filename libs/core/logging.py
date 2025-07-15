# libs/core/logging.py

import logging
import os
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


def setup_logger(name: str = "sentenialx", level: str = "INFO") -> logging.Logger:
    os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        # Console Handler (colored if available)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(_get_colored_formatter() if COLORLOG_AVAILABLE else _get_plain_formatter())
        logger.addHandler(console_handler)

        # File Handler (plain text)
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
            "DEBUG":    "cyan",
            "INFO":     "green",
            "WARNING":  "yellow",
            "ERROR":    "red",
            "CRITICAL": "bold_red"
        }
    )


# Test: only runs when executing directly
if __name__ == "__main__":
    log = setup_logger("sentenialx.test", "DEBUG")
    log.debug("Debug message")
    log.info("Info message")
    log.warning("Warning message")
    log.error("Error message")
    log.critical("Critical message")

