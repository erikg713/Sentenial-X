# apps/dashboard/utils/logger.py

"""
Dashboard Logger Utility
------------------------
Provides a centralized logger for the Sentenial-X dashboard frontend/backend integration.
"""

import logging
from logging import Logger

def init_logger(name: str = "dashboard", level: str = "INFO") -> Logger:
    """
    Initialize and return a logger instance.
    
    Args:
        name (str): Logger name.
        level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
    
    Returns:
        Logger: Configured Python logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    return logger


# Example usage
if __name__ == "__main__":
    log = init_logger()
    log.debug("Debug message")
    log.info("Info message")
    log.warning("Warning message")
    log.error("Error message")
    log.critical("Critical message")
