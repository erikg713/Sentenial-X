"""
Sentenial-X Reporting Package
-----------------------------
Provides tools for building and uploading reports
from threat monitoring, forensic logs, and analytics.

Modules:
- report_builder.py : Build structured reports in multiple formats.
- upload.py         : Save/upload reports to local or remote destinations.
"""

import logging

from .report_builder import ReportBuilder
from .upload import ReportUploader

__all__ = ["ReportBuilder", "ReportUploader"]

# Version for this reporting package
__version__ = "1.0.0"

# Configure a default logger for the reporting subsystem
logger = logging.getLogger("sentenial.reporting")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)