import logging
import os
import sys
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    Logs are serialized as JSON for easier ingestion by SIEM, ELK, or Splunk.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record)


def get_logger(name: str = "sentenialx", log_level: str = "INFO") -> logging.Logger:
    """
    Creates and configures a logger with JSON formatting, console, and file rotation.
    
    Args:
        name (str): Logger name.
        log_level (str): Minimum log level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
    
    Returns:
        logging.Logger: Configured logger instance.
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Prevent duplicate handlers

    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{name}.log")

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(JSONFormatter())

    # Console handler (human-readable)
    console_handler = logging.StreamHandler(sys.stdout)
    console_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)

    # Set log level
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    logger.info("Logger initialized successfully.")

    return logger


# Example usage
if __name__ == "__main__":
    log = get_logger("sentenialx", "DEBUG")
    log.debug("Debug message for tracing execution flow.")
    log.info("Information log for operations.")
    log.warning("Warning log for potential issues.")
    log.error("Error log with stack trace.", exc_info=True)
