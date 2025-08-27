"""
Test suite for the Semantic Analyzer core.

This package contains unit tests, integration tests, and regression tests
for all semantic analysis modules. Tests are designed to be run with
pytest or unittest, ensuring robustness and reliability of the system.
"""

import os
import sys

# Ensure the semantic_analyzer package is importable when running tests directly
PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)
