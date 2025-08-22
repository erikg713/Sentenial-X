"""
Sentenial‑X config/config.py
Centralized configuration loader and validator.
"""

import os
from pathlib import Path
import yaml

# Path to default config
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"

def load_defaults():
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def apply_env_overrides(config):
    # Example: override DB URL if set in environment
    if "DB_URL" in os.environ:
        config["database"]["url"] = os.environ["DB_URL"]
    return config

def validate(config):
    required_keys = ["database", "security", "ml"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")

# Public API
def get_config():
    cfg = load_defaults()
    cfg = apply_env_overrides(cfg)
    validate(cfg)
    return cfg

CONFIG = get_config()
