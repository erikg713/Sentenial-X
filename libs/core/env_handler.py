"""
libs/core/env_handler.py

Environment variable handler for Sentenial-X platform.
Provides type-safe access, default values, and logging for environment configuration.
"""

import os
import logging
from typing import Any, Optional

# Setup logger
logger = logging.getLogger("sentenialx.env_handler")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class EnvHandler:
    """
    Provides safe access to environment variables with type casting and defaults.
    """

    @staticmethod
    def get(key: str, default: Optional[Any] = None, required: bool = False) -> str:
        value = os.getenv(key, default)
        if required and value is None:
            logger.warning("Required environment variable '%s' not set!", key)
        return value

    @staticmethod
    def get_int(key: str, default: Optional[int] = None, required: bool = False) -> int:
        value = EnvHandler.get(key, default, required)
        try:
            return int(value)
        except (TypeError, ValueError):
            logger.warning("Environment variable '%s' is not an integer. Using default: %s", key, default)
            return default

    @staticmethod
    def get_float(key: str, default: Optional[float] = None, required: bool = False) -> float:
        value = EnvHandler.get(key, default, required)
        try:
            return float(value)
        except (TypeError, ValueError):
            logger.warning("Environment variable '%s' is not a float. Using default: %s", key, default)
            return default

    @staticmethod
    def get_bool(key: str, default: Optional[bool] = False, required: bool = False) -> bool:
        value = EnvHandler.get(key, default, required)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)


# Singleton instance for global use
env = EnvHandler()


# Example usage
if __name__ == "__main__":
    print("API_KEY:", env.get("API_KEY", "default-key"))
    print("PORT:", env.get_int("PORT", 8000))
    print("DEBUG_MODE:", env.get_bool("DEBUG_MODE", True))
