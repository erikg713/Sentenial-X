"""
Utility subpackage for the Semantic Analyzer core.

This package provides helper functions, constants, and shared utilities
that are used across the semantic analysis modules:
- Logging setup
- Text preprocessing
- Configuration management
- Common data structures

Modules are imported here for convenient access.
"""

from .logging import get_logger
from .text import normalize_text, tokenize
from .config import load_config

__all__ = [
    "get_logger",
    "normalize_text",
    "tokenize",
    "load_config",
]
