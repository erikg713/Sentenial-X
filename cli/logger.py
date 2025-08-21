# cli/logger.py
import logging
from logging.handlers import RotatingFileHandler
from .config import LOG_FILE, LOG_LEVEL

def setup_logger(name: str = "sentenial") -> logging.Logger:
    """
    Sets up a rotating file + console logger for the CLI/agent.
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(LOG_LEVEL)
    ch_formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    ch.setFormatter(ch_formatter)

    # File handler with rotation
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=5)
    fh.setLevel(LOG_LEVEL)
    fh_formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    fh.setFormatter(fh_formatter)

    # Add handlers
    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger
