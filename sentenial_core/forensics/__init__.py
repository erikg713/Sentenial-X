"""
SentenialX Forensics Module

This package provides forensic utilities and analysis tools for SentenialX A.I.
Submodules should be imported here to make them available at the package level.
"""

import logging

# Submodule imports (add/remove as needed)
from . import analyzer
from . import reporter
from . import utils  # Example: add your actual submodules

__all__ = [
    "analyzer",
    "reporter",
    "utils",   # Update this list as your submodules change
]

# Configure logging for the forensics package
logger = logging.getLogger("sentenial_core.forensics")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s][Forensics] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

logger.debug("Forensics package initialized. Submodules: %s", ', '.join(__all__))

# Optional: quick runtime check for submodule import errors
for name in __all__:
    try:
        globals()[name]
    except Exception as exc:
        logger.warning(f"Submodule '{name}' could not be imported: {exc}")

