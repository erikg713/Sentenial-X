"""
Sentenial-X ML Library
----------------------
Provides machine learning components for threat detection, NLP analysis,
distillation, and model fine-tuning.

Submodules:
- encoder      : Threat text/vector encoders
- distill      : Distilled student models for lightweight inference
- lora         : LoRA adapters for fine-tuning
- utils        : Shared ML utilities
"""

import logging

# Import submodules for easy access
from .encoder.text_encoder import ThreatTextEncoder
from .distill.student_model import DistilledThreatModel
from .lora.lora_loader import load_lora_model
from .lora.lora_tuner import LoRATuner
from . import utils

__all__ = [
    "ThreatTextEncoder",
    "DistilledThreatModel",
    "load_lora_model",
    "LoRATuner",
    "utils",
]

# Version of ML library
__version__ = "1.0.0"

# Configure default logger
logger = logging.getLogger("sentenial.ml")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
