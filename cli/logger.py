"""
cli/logger.py

Centralized logging module for Sentenial-X CLI.

Features:
- Structured JSON logging for easy parsing
- Console and optional file logging
- Supports multiple modules (Cortex, WormGPT, Telemetry, Alerts, etc.)
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime

# ------------------------------
# Formatter Classes
# ------------------------------
class JSONFormatter(logging.Formatter):
    """Outputs logs in JSON format for structured logging."""

    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record)


# ------------------------------
# Logger Setup Function
# ------------------------------
def get_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured logger instance.

    Args:
        name (str): Module or application name
        log_file (str, optional): Path to a log file
        level (int): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # prevent duplicate logs in root

    # Remove existing handlers to avoid duplicate logs on multiple calls
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = JSONFormatter()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler with rotation (optional)
    if log_file:
        file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
        file_handler.setLevel(level)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

    return logger


# ------------------------------
# Default logger instance
# ------------------------------
# Use this default logger in modules if no custom name is needed
default_logger = get_logger("SentenialX", log_file="sentenialx.log", level=logging.INFO)


# ------------------------------
# Quick helper functions
# ------------------------------
def log_info(msg: str):
    default_logger.info(msg)


def log_debug(msg: str):
    default_logger.debug(msg)


def log_warn(msg: str):
    default_logger.warning(msg)


def log_error(msg: str):
    default_logger.error(msg)


def log_exception(msg: str):
    default_logger.exception(msg)


# ------------------------------
# Optional test run
# ------------------------------
if __name__ == "__main__":
    logger = get_logger("TestLogger", log_file="test.log", level=logging.DEBUG)
    logger.info("Test info message")
    logger.debug("Debug message with details")
    logger.warning("Warning message")
    logger.error("Error message")
    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception("Caught division by zero")
