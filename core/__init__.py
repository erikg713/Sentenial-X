# core/__init__.py
"""
Core package initialization for Sentenial-X.

Responsibilities:
- Provide a package-level logger and a helper to configure logging consistently.
- Expose get_version() to discover the installed package version (best-effort).
- Provide a small lazy-import mechanism so commonly used submodules/components can
  be accessed as attributes without forcing imports at package import time.
- Attempt optional imports of well-known components (e.g. Dashboard) but never
  raise if they are absent; keep package import safe.

The style prefers explicit, readable code and minimal runtime work during import.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Optional

# Try to import importlib.metadata in a way that works across supported Python versions
try:
    # Python 3.8+
    from importlib import metadata as importlib_metadata  # type: ignore
except Exception:
    try:
        # Backport
        import importlib_metadata  # type: ignore
    except Exception:
        importlib_metadata = None  # type: ignore

__all__ = [
    "logger",
    "configure_logging",
    "get_version",
]

logger = logging.getLogger(__name__)
# Note: we do not configure global logging by default; callers should call configure_logging.

def configure_logging(level: int = logging.INFO, fmt: Optional[str] = None) -> None:
    """
    Configure a sensible default logging setup for modules that use core.logger.

    Parameters:
    - level: logging level (defaults to INFO)
    - fmt: optional format string for the handler; a reasonable default is used when omitted.
    """
    if fmt is None:
        fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    # Only add a handler if none exist to avoid duplicate logs in interactive sessions
    if not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(level)


def get_version(package_names: Optional[list[str]] = None) -> Optional[str]:
    """
    Return an installed distribution version for this project (best-effort).

    The function attempts a small list of likely distribution names. If none are
    found or importlib.metadata isn't available, returns None.
    """
    candidates = package_names or ["sentenial-x", "sentenial_x", "sentenialx", "Sentenial-X"]
    if importlib_metadata is None:
        logger.debug("importlib.metadata not available; cannot determine package version.")
        return None

    for name in candidates:
        try:
            ver = importlib_metadata.version(name)
            logger.debug("Found package version %s for distribution %s", ver, name)
            return ver
        except importlib_metadata.PackageNotFoundError:
            continue
        except Exception as exc:
            # Unexpected error; log at debug to aid troubleshooting but don't raise.
            logger.debug("Error while querying package version for %s: %s", name, exc)
    logger.debug("No matching distribution found from candidates: %s", candidates)
    return None


# Lazy import helpers:
# Map friendly attribute names -> module paths (or callables that resolve)
_lazy_map: dict[str, Callable[[], Any]] = {}


def _make_lazy(module_path: str, attr: Optional[str] = None) -> Callable[[], Any]:
    def _loader() -> Any:
        mod = importlib.import_module(module_path)
        return getattr(mod, attr) if attr else mod
    return _loader


# Commonly used top-level accessors (adjust names to your project's real modules)
# These are attempted lazily when accessed via `from core import Dashboard` or `core.Dashboard`.
_lazy_map.update(
    {
        "Dashboard": _make_lazy(__name__ + ".dashboard", "Dashboard"),
        "components": _make_lazy(__name__ + ".components"),
        "utils": _make_lazy(__name__ + ".utils"),
        "cli": _make_lazy(__name__ + ".cli"),
        "api": _make_lazy(__name__ + ".api"),
    }
)


def __getattr__(name: str) -> Any:
    """
    Module-level getattr hook for lazy attributes.

    If `name` is present in _lazy_map we import on demand; otherwise raise AttributeError
    to mimic normal attribute access behavior.
    """
    loader = _lazy_map.get(name)
    if loader is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        value = loader()
        # Cache the loaded value on the module to avoid repeated imports.
        globals()[name] = value
        # If it is an attribute we also expose it in __all__
        if name not in __all__:
            __all__.append(name)
        return value
    except Exception as exc:
        # Wrap import errors to give a clearer runtime message without losing context.
        logger.debug("Lazy import for %s failed: %s", name, exc, exc_info=True)
        raise ImportError(f"Could not import {name!r} from {__name__!r}") from exc


def __dir__() -> list[str]:
    # Combine explicit exports and lazy-map keys for tab-completion friendliness.
    names = set(__all__) | set(_lazy_map.keys()) | set(globals().keys())
    return sorted(names)


# Attempt to import a few commonly referenced objects at import time if present.
# This is optional and silent: we want to avoid hard failures if those modules don't exist.
_optional_try_imports = {
    "Dashboard": ("dashboard", "Dashboard"),
    # Add other optional eager imports here if you want them available immediately.
}

for public_name, (mod_name, attr_name) in _optional_try_imports.items():
    try:
        mod = importlib.import_module(f"{__name__}.{mod_name}")
        obj = getattr(mod, attr_name)
        globals()[public_name] = obj
        if public_name not in __all__:
            __all__.append(public_name)
    except Exception:
        # Intentionally ignore errors — keep import-time side effects minimal.
        logger.debug("Optional import %s.%s not available.", mod_name, attr_name)


# Expose package version at import time (best-effort) as a convenient constant.
__version__ = get_version() or "0.0.0+local"
__all__.append("__version__")
