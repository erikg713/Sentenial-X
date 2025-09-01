# agents/config.py
"""
Sentenial-X Agent Configuration
--------------------------------
Centralized configuration for endpoint, network, and other modular agents.

This file ensures all agents share consistent settings for logging, 
communication, update intervals, and security keys.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any

# ==============================
# Global Paths & Directories
# ==============================
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# Logging Configuration
# ==============================
LOG_LEVEL = os.getenv("SENTENIAL_AGENT_LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    filename=LOG_DIR / "agent.log",
    filemode="a",
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=getattr(logging, LOG_LEVEL, logging.INFO),
)

logger = logging.getLogger("SentenialX.Agent")

# ==============================
# Agent Runtime Config
# ==============================
AGENT_CONFIG: Dict[str, Any] = {
    "heartbeat_interval": int(os.getenv("SENTENIAL_HEARTBEAT_INTERVAL", "10")),  # seconds
    "max_retry_attempts": int(os.getenv("SENTENIAL_MAX_RETRY", "5")),
    "secure_channel": os.getenv("SENTENIAL_SECURE_CHANNEL", "wss://localhost:9443/agent"),
    "auth_token": os.getenv("SENTENIAL_AGENT_AUTH_TOKEN", "changeme"),
    "agent_id": os.getenv("SENTENIAL_AGENT_ID", "default-agent"),
    "data_buffer_limit": int(os.getenv("SENTENIAL_BUFFER_LIMIT", "5000")),  # max events in buffer
}

# ==============================
# Security Config
# ==============================
SECURITY_KEYS = {
    "encryption_key": os.getenv("SENTENIAL_ENCRYPTION_KEY", "default-key"),
    "hmac_secret": os.getenv("SENTENIAL_HMAC_SECRET", "default-hmac"),
}

# ==============================
# Utility Functions
# ==============================
def get_config() -> Dict[str, Any]:
    """
    Returns a deep copy of agent config to prevent accidental mutations.
    """
    return {**AGENT_CONFIG}


def reload_logging(level: str = None):
    """
    Reloads logging with a new level dynamically.
    """
    new_level = level or LOG_LEVEL
    logger.setLevel(getattr(logging, new_level, logging.INFO))
    logger.info(f"Logging level updated to {new_level}")


logger.info("Agent configuration initialized successfully.")
