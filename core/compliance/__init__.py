"""
Core compliance package — small, stable public surface for legal ontology parsing.

This module intentionally keeps a compact, stable API:
- parse_policy(...) : primary entrypoint (thin, robust proxy to the implementation)
- load_policy_from_file(path, encoding='utf-8') : convenience helper to parse a policy file
- load_policy_from_text(text) : convenience helper when you already have text

The implementation import is guarded and reported with a helpful error if the
parser cannot be imported to avoid confusing import-time tracebacks elsewhere in the app.
"""
from __future__ import annotations

from typing import Any, Optional
import logging
import io

_logger = logging.getLogger(__name__)

# Try to obtain a package version (best-effort). Fall back to a sensible local default.
try:
    # Use the top-level package name (likely "core" or project root). This is best-effort only.
    from importlib.metadata import version as _get_version, PackageNotFoundError

    try:
        __version__ = _get_version(__name__.split(".")[0])
    except PackageNotFoundError:
        __version__ = "0+local"
except Exception:
    __version__ = "0+local"

# Guarded import: keep the module import cheap and provide a friendly runtime error if the
# implementation can't be loaded (helps with tooling and when importing just for metadata).
_parse_impl = None
_import_error: Optional[BaseException] = None
try:
    from .legal_ontology_parser import parse_policy as _parse_impl  # type: ignore
except Exception as exc:  # pragma: no cover - defensive import-time handling
    _import_error = exc
    _logger.debug("Failed to import legal_ontology_parser at package import time: %s", exc)


def parse_policy(*args: Any, **kwargs: Any) -> Any:
    """
    Parse a legal policy representation.

    This is a thin, backward-compatible proxy to the real implementation in
    core.compliance.legal_ontology_parser.parse_policy. The wrapper delays and
    centralizes the import error handling so callers get a clear, actionable
    message if something went wrong during import.

    Any args/kwargs are forwarded to the underlying implementation.
    """
    if _parse_impl is None:
        msg = (
            "core.compliance.legal_ontology_parser.parse_policy is not available. "
            "Ensure the module core.compliance.legal_ontology_parser can be imported. "
        )
        if _import_error is not None:
            # Provide the original error for faster debugging while keeping the message concise.
            raise ImportError(msg) from _import_error
        raise ImportError(msg)
    return _parse_impl(*args, **kwargs)


def load_policy_from_file(path: str, encoding: str = "utf-8") -> Any:
    """
    Read a policy file and parse it.

    - path: filesystem path to the policy file
    - encoding: text encoding to use when opening the file (default: 'utf-8')

    Returns whatever parse_policy(...) returns (commonly a dict / structured object).
    """
    # Use io.open for consistent behaviour across Python versions
    _logger.debug("Loading policy from file: %s", path)
    with io.open(path, "r", encoding=encoding) as fh:
        content = fh.read()
    try:
        return parse_policy(content)
    except Exception as exc:
        _logger.exception("Failed to parse policy file %s", path)
        raise


def load_policy_from_text(text: str) -> Any:
    """
    Parse a policy from a text string and return the parsed representation.

    Convenience wrapper around parse_policy for clearer call sites.
    """
    _logger.debug("Parsing policy from text input (len=%d)", len(text) if text is not None else 0)
    return parse_policy(text)


__all__ = [
    "parse_policy",
    "load_policy_from_file",
    "load_policy_from_text",
    "__version__",
  ]
from .legal_ontology_parser import parse_policy
