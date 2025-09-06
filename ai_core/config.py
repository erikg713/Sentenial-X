# -*- coding: utf-8 -*-
"""
AI Core Configuration for Sentenial-X
--------------------------------------

Centralizes configuration for AI models, thresholds, and runtime
settings. Supports environment variable overrides.
"""

from __future__ import annotations

import os
from typing import Literal

# ---------------------------------------------------------------------------
# General AI Core Settings
# ---------------------------------------------------------------------------

LOG_LEVEL: str = os.getenv("SENTENIALX_AI_LOG_LEVEL", "INFO").upper()
MODEL_PATH_NLP: str = os.getenv("SENTENIALX_MODEL_PATH_NLP", "models/nlp_model.pt")
MODEL_PATH_ADVERSARIAL: str = os.getenv("SENTENIALX_MODEL_PATH_ADVERSARIAL", "models/adversarial_model.pt")
MODEL_PATH_PREDICTIVE: str = os.getenv("SENTENIALX_MODEL_PATH_PREDICTIVE", "models/predictive_model.pt")

# ---------------------------------------------------------------------------
# Analyzer thresholds
# ---------------------------------------------------------------------------

# NLP Analyzer thresholds
NLP_CONFIDENCE_THRESHOLD: float = float(os.getenv("SENTENIALX_NLP_CONF_THRESHOLD", 0.75))

# Adversarial Detector thresholds
ADVERSARIAL_CONFIDENCE_THRESHOLD: float = float(os.getenv("SENTENIALX_ADV_CONF_THRESHOLD", 0.8))

# Predictive Threat Model thresholds
PREDICTIVE_RISK_THRESHOLD: float = float(os.getenv("SENTENIALX_PREDICTIVE_RISK_THRESHOLD", 0.7))

# ---------------------------------------------------------------------------
# Misc Settings
# ---------------------------------------------------------------------------

# Allowed log levels
ALLOWED_LOG_LEVELS: list[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]] = [
    "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
]

# Random seed for reproducibility
RANDOM_SEED: int = int(os.getenv("SENTENIALX_AI_RANDOM_SEED", 42))

# Enable GPU acceleration if available
USE_GPU: bool = os.getenv("SENTENIALX_USE_GPU", "False").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Helper function to validate log level
# ---------------------------------------------------------------------------
def get_valid_log_level(level: str) -> str:
    """
    Ensure the provided log level is valid; fallback to INFO.
    """
    level = level.upper()
    if level not in ALLOWED_LOG_LEVELS:
        return "INFO"
    return level
