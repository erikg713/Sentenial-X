"""
Sentenial‑X Configuration Package
Provides a unified interface for loading and validating settings.
"""
# config/__init__.py

"""
Sentenial-X Configuration Package
---------------------------------
Centralized configuration module for API, CLI, AI core, and dashboard components.
"""

from .settings import Settings
from pathlib import Path
import yaml

# Load defaults from YAML
_config_path = Path(__file__).parent / "config.yaml.example"

with open(_config_path, "r", encoding="utf-8") as f:
    DEFAULTS = yaml.safe_load(f)

# Optional: environment‑specific overrides
try:
    from .local_settings import LOCAL_OVERRIDES
    DEFAULTS.update(LOCAL_OVERRIDES)
except ImportError:
    pass

# Public API
__all__ = ["DEFAULTS"]
