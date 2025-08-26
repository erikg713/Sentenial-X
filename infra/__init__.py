"""
Sentenial-X Infrastructure Layer
================================

Provides low-level infrastructure components such as:
- Databases
- Caching
- Queues / Brokers
- Configuration

This layer abstracts implementation details to support higher-level
modules like Cortex, Semantic Analyzer, and Models.
"""

import logging

__version__ = "0.1.0"

# Setup base logger
logger = logging.getLogger("sentenial_x.infra")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] infra :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

logger.debug("Initializing Sentenial-X Infrastructure Layer")

# Public API exposure
from . import db, cache, queue, config

__all__ = [
    "__version__",
    "logger",
    "db",
    "cache",
    "queue",
    "config",
]
