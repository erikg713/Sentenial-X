"""
libs/core/config_loader.py

Centralized configuration loader for Sentenial-X platform.
Supports YAML and JSON configuration files, environment variable overrides,
and provides a unified dictionary interface for all modules.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Union
import logging

# Setup logger
logger = logging.getLogger("sentenialx.config_loader")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ConfigLoader:
    """
    Loads and manages configuration for Sentenial-X modules.
    Supports YAML and JSON config formats with environment variable overrides.
    """

    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self):
        if not self.config_path.exists():
            logger.warning("Config file not found: %s", self.config_path)
            self.config = {}
            return

        try:
            if self.config_path.suffix in [".yaml", ".yml"]:
                with open(self.config_path, "r") as f:
                    self.config = yaml.safe_load(f) or {}
            elif self.config_path.suffix == ".json":
                with open(self.config_path, "r") as f:
                    self.config = json.load(f)
            else:
                logger.error("Unsupported config format: %s", self.config_path.suffix)
                self.config = {}

            logger.info("Loaded configuration from %s", self.config_path)
        except Exception as e:
            logger.exception("Failed to load configuration: %s", e)
            self.config = {}

        # Apply environment variable overrides
        for key in self.config.keys():
            env_val = os.getenv(key.upper())
            if env_val is not None:
                logger.info("Overriding config key %s with environment variable", key)
                self.config[key] = env_val

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def all(self) -> Dict[str, Any]:
        return self.config.copy()


# Singleton loader (optional)
def load_config(config_path: Union[str, Path] = None) -> ConfigLoader:
    config_file = config_path or os.getenv("SENTENIALX_CONFIG", "config/config.yaml")
    return ConfigLoader(config_file)


# Example usage
if __name__ == "__main__":
    loader = load_config()
    print("Full config:", loader.all())
    print("API_KEY:", loader.get("API_KEY", "not_set"))
