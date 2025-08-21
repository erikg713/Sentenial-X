# -*- coding: utf-8 -*-
"""
Package initializer for the `scripts` helper package.

This module provides:
- a lightweight, robust way to obtain the package version
- a small, opinionated logging setup used by scripts in this package
- utilities to discover and run submodules in the `scripts` package
The code favours explicitness, sensible defaults, and clear error messages.
"""
from __future__ import annotations

import importlib
import logging
import os
import pkgutil
from typing import Any, Callable, Iterable, List, Optional

# importlib.metadata is available in stdlib for Python >=3.8
try:
    from importlib import metadata as _metadata  # type: ignore
except Exception:  # pragma: no cover - extremely old Python
    _metadata = None  # type: ignore

# Public API
__all__ = [
    "__version__",
    "configure_logging",
    "get_logger",
    "discover_scripts",
    "run_script",
]

# Internal state to avoid configuring logging multiple times
_LOGGING_CONFIGURED = False


def _guess_distribution_names() -> Iterable[str]:
    """
    Generator of plausible distribution names for this project.
    Add more variations here if your packaging uses a different name.
    """
    yield "Sentenial-X"
    yield "sentenial-x"
    yield "sentenial_x"
    yield "sentenialx"
    # falling back to the package folder name
    yield __name__.split(".")[0]


def _get_version_from_metadata() -> str:
    """
    Try to obtain the package version from installed metadata.
    Returns "0.0.0" if not available (e.g., running from source tree).
    """
    if _metadata is None:
        return "0.0.0"
    for name in _guess_distribution_names():
        try:
            return _metadata.version(name)
        except Exception:
            continue
    return "0.0.0"


# Expose a stable version string even when the package is not installed
__version__ = _get_version_from_metadata()


def configure_logging(level: Optional[int | str] = None, *, force: bool = False) -> None:
    """
    Configure package logging with a concise, readable default format.

    - level: optional logging level (int or str). If omitted, falls back to
      the SENTENIALX_LOGLEVEL environment variable, or INFO.
    - force: when True, reconfigure logging even if it was already configured.

    This function is idempotent by default; call with force=True to override an
    existing configuration (useful in tests).
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED and not force:
        return

    env_level = os.getenv("SENTENIALX_LOGLEVEL")
    chosen_level = level if level is not None else env_level or "INFO"

    if isinstance(chosen_level, str):
        chosen_level = chosen_level.upper()
        resolved_level = getattr(logging, chosen_level, None)
        if resolved_level is None:
            try:
                resolved_level = int(chosen_level)
            except Exception:
                resolved_level = logging.INFO
    else:
        resolved_level = int(chosen_level)

    # Use a single StreamHandler with a terse formatter; keep it simple and deterministic.
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    # If not forcing, avoid duplicating handlers
    if force:
        for h in list(root.handlers):
            root.removeHandler(h)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)

    root.setLevel(resolved_level)
    _LOGGING_CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a configured logger for `name`. If logging isn't configured yet,
    this will configure it using defaults.

    Example:
        log = get_logger(__name__)
        log.info("Started")
    """
    if not _LOGGING_CONFIGURED:
        configure_logging()
    return logging.getLogger(name or "sentenialx")


def discover_scripts() -> List[str]:
    """
    Discover top-level modules inside this package (non-private).

    Returns a sorted list of module names (strings) that can be imported as
    `scripts.<name>`. Modules and packages whose names start with an underscore
    are ignored.
    """
    modules = []
    for finder, name, ispkg in pkgutil.iter_modules(path=__path__, prefix=""):
        if name.startswith("_"):
            continue
        modules.append(name)
    modules.sort()
    return modules


def run_script(name: str, *args: Any, entry: Optional[str] = None, **kwargs: Any) -> Any:
    """
    Import and run a script module by name.

    Behavior:
    - Imports module `scripts.<name>`.
    - If `entry` is provided, it will call that attribute from the module (must be callable).
    - Otherwise, it prefers to call a callable attribute named "main", then "run".
    - If no suitable callable is found, raises RuntimeError.

    Returns whatever the callable returns.

    Example:
        run_script("clean_cache", "--dry-run")
    """
    if not name or not isinstance(name, str):
        raise TypeError("name must be a non-empty string")

    module_name = f"{__name__}.{name}"
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise ImportError(f"Could not import script module '{module_name}': {exc}") from exc

    candidate: Optional[Callable[..., Any]] = None
    if entry:
        candidate = getattr(module, entry, None)
        if not callable(candidate):
            raise RuntimeError(f"Module '{module_name}' has no callable '{entry}' entrypoint")
    else:
        for attr in ("main", "run"):
            candidate = getattr(module, attr, None)
            if callable(candidate):
                break

    if not callable(candidate):
        raise RuntimeError(
            f"Module '{module_name}' does not expose a callable entrypoint (expected 'main' or 'run')"
        )

    # It's a convenience wrapper; the module's callable is responsible for arg parsing.
    return candidate(*args, **kwargs)
