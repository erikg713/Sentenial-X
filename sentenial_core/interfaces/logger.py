# sentenial_core/interfaces/logger.py

from loguru import logger
import sys
from sentenial_core.interfaces.config import config

def setup_logger():
    # Remove default handler
    logger.remove()

    # Configure logger with dynamic level and formatting
    logger.add(
        sys.stdout,
        level=config.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        backtrace=True,
        diagnose=True,
        enqueue=True,
        colorize=True,
    )

    # Optionally, add file logging
    # logger.add("logs/sentenial_{time:YYYY-MM-DD}.log", rotation="10 MB", retention="7 days")

# Initialize logger on import
setup_logger()

# Usage:
# from sentenial_core.interfaces.logger import logger
# logger.info("This is an info message")

