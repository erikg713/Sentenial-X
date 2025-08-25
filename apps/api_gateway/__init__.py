"""""""python
"""
Small utilities and package-level exports for the api_gateway app.

This module provides:
- package metadata (__version__, __author__)
- a lightweight, dependency-free SimpleGateway dataclass for building API URLs and headers
- helper functions: get_version(package_name), configure_logger(level)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List

# Prefer importlib.metadata (stdlib) with a compatible backport fallback
try:
    from importlib.metadata import version as _get_version  # type: ignore
except Exception:
    try:
        from importlib_metadata import version as _get_version  # type: ignore
    except Exception:
        _get_version = None  # type: ignore

__all__ = ["SimpleGateway", "get_version", "configure_logger", "__version__"]

logger = logging.getLogger(__name__)

def configure_logger(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the package logger.

    This function is idempotent: calling it multiple times won't add duplicate handlers.
    """
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s - %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

def get_version(package_name: str) -> str:
    """Return the installed distribution version for package_name or '0.0.0' if unknown.

    The function intentionally swallows exceptions and returns a stable fallback so imports
    never fail in environments where distribution metadata is not available.
    """
    if _get_version is None:
        logger.debug("importlib.metadata not available; returning fallback version")
        return "0.0.0"
    try:
        return _get_version(package_name)
    except Exception:
        logger.debug("failed to get version for %s", package_name, exc_info=True)
        return "0.0.0"

@dataclass(frozen=True)
class SimpleGateway:
    """Lightweight helper for building API endpoint URLs and default headers.

    This class intentionally avoids making network calls so it can be used in any
    runtime (including unit tests) without pulling in extra HTTP client dependencies.
    """
    base_url: str
    default_headers: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: Optional[int] = 30

    def build_url(self, path: str) -> str:
        """Return a normalized URL joining base_url and path.

        Examples:
            gw = SimpleGateway("https://api.example.com/v1")
            gw.build_url("users") -> "https://api.example.com/v1/users"
        """
        if not path:
            return self.base_url
        if path.startswith("/"):
            path = path[1:]
        if not self.base_url.endswith("/"):
            return f"{self.base_url}/{path}"
        return f"{self.base_url}{path}"

    def headers_for(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Return merged headers where 'extra' overrides defaults."""
        headers = dict(self.default_headers)
        if extra:
            headers.update(extra)
        return headers

    def with_auth(self, token: str, scheme: str = "Bearer") -> "SimpleGateway":
        """Return a shallow copy with Authorization header set (immutably).

        Because the dataclass is frozen, this constructs and returns a new instance.
        """
        new_headers = dict(self.default_headers)
        new_headers["Authorization"] = f"{scheme} {token}"
        return SimpleGateway(base_url=self.base_url, default_headers=new_headers, timeout_seconds=self.timeout_seconds)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"SimpleGateway(base_url={self.base_url!r}, headers={len(self.default_headers)} items, timeout={self.timeout_seconds!r})"

# package metadata: attempt a few sensible distribution names and fall back to 0.0.0
__version__ = "0.0.0"
_candidates: List[str] = []
if __package__:
    _c = __package__.split(".")[0]
    if _c:
        _candidates.append(_c)
# common variations of the repository name
_candidates.extend(["Sentenial-X", "sentenial_x", "sentenial-x", "sentenialx"])  # best-effort
for _cand in _candidates:
    try:
        v = get_version(_cand)
    except Exception:
        v = "0.0.0"
    if v and v != "0.0.0":
        __version__ = v
        break

__author__ = "erikg713"

# Configure a conservative default log level to avoid noisy output on import
configure_logger(logging.WARNING)
"""