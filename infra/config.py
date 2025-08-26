"""
Configuration Loader
====================

Loads configuration from environment variables or .env files.
"""

import os
from dotenv import load_dotenv
from . import logger

# Load .env if available
load_dotenv()


def get(key: str, default=None):
    """Get configuration value."""
    value = os.getenv(key, default)
    logger.debug(f"Config: {key} = {value}")
    return value


# Example commonly used configs
DATABASE_URL = get("DATABASE_URL", "sqlite:///sentenial.db")
REDIS_URL = get("REDIS_URL", "redis://localhost:6379/0")
KAFKA_BROKER = get("KAFKA_BROKER", "localhost:9092")
