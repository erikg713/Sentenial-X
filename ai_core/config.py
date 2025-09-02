"""
Sentenial-X AI Core Configuration Module
----------------------------------------
Handles centralized configuration for AI/ML pipelines, models,
and AI Core services. Supports environment overrides, secure secret
loading, and structured access to model parameters.

Author: Sentenial-X Development Team
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

logger = logging.getLogger("SentenialX.AICore.Config")
logger.setLevel(logging.DEBUG)


class AIConfig:
    """
    AI Core configuration manager.
    Provides structured access to AI model, pipeline, and runtime settings.
    """

    def __init__(self, config_file: Optional[str] = None):
        self.base_dir = Path(__file__).resolve().parent
        self.config_file = config_file or os.getenv("AI_CORE_CONFIG", str(self.base_dir / "config.json"))
        self.config: Dict[str, Any] = {}

        self._load_config()

    def _load_config(self):
        """Load configuration from file and environment variables."""
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
                logger.info(f"Loaded AI Core config from {self.config_file}")
            else:
                logger.warning(f"No config file found at {self.config_file}, using environment variables only")

            # Merge environment variables
            self._apply_env_overrides()

        except Exception as e:
            logger.error(f"Failed to load AI Core config: {e}")
            self.config = {}

    def _apply_env_overrides(self):
        """Override configuration values with environment variables when available."""
        overrides = {
            "MODEL_PATH": os.getenv("AI_MODEL_PATH"),
            "MODEL_TYPE": os.getenv("AI_MODEL_TYPE"),
            "CACHE_DIR": os.getenv("AI_CACHE_DIR"),
            "GPU_ENABLED": os.getenv("AI_GPU_ENABLED"),
            "MAX_BATCH_SIZE": os.getenv("AI_MAX_BATCH_SIZE"),
        }

        for key, value in overrides.items():
            if value is not None:
                self.config[key] = self._cast_value(value)

    @staticmethod
    def _cast_value(value: str) -> Any:
        """Cast environment variable strings into appropriate Python types."""
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        if value.isdigit():
            return int(value)
        try:
            return float(value)
        except ValueError:
            return value

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a configuration value with a default fallback."""
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        """Set a configuration value dynamically (runtime overrides)."""
        self.config[key] = value

    def save(self):
        """Persist current configuration back to the config file."""
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    # --- Model-Specific Helpers ---

    def get_model_path(self) -> str:
        """Get AI model path (local or remote)."""
        return self.get("MODEL_PATH", str(self.base_dir / "models" / "default_model"))

    def get_model_type(self) -> str:
        """Get AI model type (transformer, rnn, anomaly-detector, etc.)."""
        return self.get("MODEL_TYPE", "transformer")

    def gpu_enabled(self) -> bool:
        """Check if GPU acceleration is enabled."""
        return bool(self.get("GPU_ENABLED", False))

    def max_batch_size(self) -> int:
        """Get maximum batch size for inference."""
        return int(self.get("MAX_BATCH_SIZE", 32))


# Singleton accessor
_ai_config: Optional[AIConfig] = None


def get_ai_config() -> AIConfig:
    """Get a singleton instance of the AI configuration."""
    global _ai_config
    if _ai_config is None:
        _ai_config = AIConfig()
    return _ai_config


if __name__ == "__main__":
    cfg = get_ai_config()
    print("AI Model Path:", cfg.get_model_path())
    print("AI Model Type:", cfg.get_model_type())
    print("GPU Enabled:", cfg.gpu_enabled())
    print("Max Batch Size:", cfg.max_batch_size())
