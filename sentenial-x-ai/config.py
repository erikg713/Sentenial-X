"""
Sentenial-X AI Configuration
============================

Central configuration module for the Sentenial-X AI framework.
Supports environment variables, default values, and structured settings.
"""

import os
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parent

class Config:
    """
    Central configuration class for Sentenial-X.
    """

    # General
    AGENT_DEFAULT_INTERVAL: float = float(os.getenv("SENTENIAL_AGENT_INTERVAL", 1.0))
    LOG_LEVEL: str = os.getenv("SENTENIAL_LOG_LEVEL", "INFO").upper()

    # CyberBattleSim
    CYBERBATTLE_ENV: str = os.getenv("SENTENIAL_CYBERBATTLE_ENV", "default")
    CYBERBATTLE_MAX_EPISODES: int = int(os.getenv("SENTENIAL_CYBERBATTLE_EPISODES", 1000))

    # Paths
    DATA_DIR: Path = Path(os.getenv("SENTENIAL_DATA_DIR", BASE_DIR / "data"))
    MODELS_DIR: Path = Path(os.getenv("SENTENIAL_MODELS_DIR", BASE_DIR / "models"))
    LOGS_DIR: Path = Path(os.getenv("SENTENIAL_LOGS_DIR", BASE_DIR / "logs"))

    # Reinforcement Learning / ML
    RL_POLICY: str = os.getenv("SENTENIAL_RL_POLICY", "DQN")
    RL_LEARNING_RATE: float = float(os.getenv("SENTENIAL_RL_LR", 0.001))
    RL_DISCOUNT_FACTOR: float = float(os.getenv("SENTENIAL_RL_GAMMA", 0.99))
    RL_BATCH_SIZE: int = int(os.getenv("SENTENIAL_RL_BATCH_SIZE", 64))

    # Multi-Agent
    MULTI_AGENT_ENABLED: bool = os.getenv("SENTENIAL_MULTI_AGENT", "False").lower() == "true"

    @classmethod
    def ensure_directories(cls):
        """Ensure that all required directories exist."""
        for path in [cls.DATA_DIR, cls.MODELS_DIR, cls.LOGS_DIR]:
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def as_dict(cls) -> Dict[str, Any]:
        """Return configuration as a dictionary."""
        return {
            "AGENT_DEFAULT_INTERVAL": cls.AGENT_DEFAULT_INTERVAL,
            "LOG_LEVEL": cls.LOG_LEVEL,
            "CYBERBATTLE_ENV": cls.CYBERBATTLE_ENV,
            "CYBERBATTLE_MAX_EPISODES": cls.CYBERBATTLE_MAX_EPISODES,
            "DATA_DIR": str(cls.DATA_DIR),
            "MODELS_DIR": str(cls.MODELS_DIR),
            "LOGS_DIR": str(cls.LOGS_DIR),
            "RL_POLICY": cls.RL_POLICY,
            "RL_LEARNING_RATE": cls.RL_LEARNING_RATE,
            "RL_DISCOUNT_FACTOR": cls.RL_DISCOUNT_FACTOR,
            "RL_BATCH_SIZE": cls.RL_BATCH_SIZE,
            "MULTI_AGENT_ENABLED": cls.MULTI_AGENT_ENABLED,
        }

# Ensure directories exist on import
Config.ensure_directories()
