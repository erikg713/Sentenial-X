"""""""""""""""""""""""""""""""""
# apps/gui_desktop/__init__.py
# Clean, human-styled package initializer for the desktop GUI app code.
# - Lazy-loads GUI backends to avoid importing heavy GUI libraries at package import time.
# - Provides utility helpers to discover and load backends safely.
# - Exposes a small, explicit public surface for downstream code.

from __future__ import annotations

"""apps.gui_desktop

This package initializer keeps imports lightweight and provides helper
utilities to discover and lazily load GUI backends. The goal is to make
importing the package cheap while still giving a clear, ergonomic API
for selecting and loading the desired GUI implementation at runtime.

Design principles used here:
- avoid importing heavy GUI libraries on module import
- provide clear error messages when backends are missing
- keep the public API small and explicit
"""

import importlib
import importlib.util
import logging
import os
import pkgutil
import sys
from typing import List, Optional, Sequence

LOGGER = logging.getLogger(__name__)

__all__ = [
    "available_backends",
    "load_backend",
    "get_version",
]

# Try to determine a sensible package version from distribution metadata.
def _find_version() -> str:
    # Prefer the distribution name if available, but fall back gracefully.
    candidates = [
        "apps.gui_desktop",
        "gui_desktop",
        "Sentenial-X",
        "sentenial-x",
    ]

    # importlib.metadata is only available in Python 3.8+ under this name; use importlib.metadata via importlib if present.
    try:
        import importlib.metadata as _metadata
    except Exception:
        _metadata = None

    if _metadata is not None:
        for name in candidates:
            try:
                return _metadata.version(name)
            except Exception:
                continue

    # If distribution metadata is unavailable, fall back to a constant or empty string.
    return "0.0.0"

__version__ = _find_version()

# Typical backend names we might look for as either submodules of this package
# or top-level importable modules. This list is intentionally conservative.
_BACKEND_CANDIDATES: Sequence[str] = (
    "qt",
    "pyqt5",
    "pyqt6",
    "pyside2",
    "pyside6",
    "gtk",
    "tkinter",
    "wx",
    "pygame",
)


def _discover_submodules() -> List[str]:
    """Return a list of submodule names bundled under this package.

    This scans the package __path__ for modules without importing heavy
    backends. The returned names are the plain module names (not full
    import paths).
    """
    found: List[str] = []
    for module_info in pkgutil.iter_modules(__path__):
        name = module_info.name
        found.append(name)
    return found


def available_backends() -> List[str]:
    """Return a list of backend names that look available.

    This includes backends that are either provided as submodules of
    apps.gui_desktop or importable top-level packages matching
    common GUI backend names.
    """
    available = set()

    # Backends implemented as submodules in this package
    for sub in _discover_submodules():
        if sub.lower() in _BACKEND_CANDIDATES:
            available.add(sub)

    # Backends that are importable from the environment
    for candidate in _BACKEND_CANDIDATES:
        try:
            if importlib.util.find_spec(candidate) is not None:
                available.add(candidate)
        except Exception:
            # Any unexpected error shouldn't break discovery; log and continue.
            LOGGER.debug("Error checking availability of %s", candidate, exc_info=True)

    # Preserve a deterministic order based on _BACKEND_CANDIDATES then submodules
    ordered: List[str] = []
    for name in _BACKEND_CANDIDATES:
        if name in available and name not in ordered:
            ordered.append(name)
    for name in sorted(available):
        if name not in ordered:
            ordered.append(name)

    return ordered


def load_backend(name: Optional[str] = None):
    """Load and return a GUI backend module.

    - If name is None, attempt to select a sensible default using
      the SENTENIAL_GUI_BACKEND environment variable or the first
      available candidate.
    - If the named backend exists as a submodule of apps.gui_desktop,
      import that. Otherwise attempt to import a top-level package
      with the given name.

    Raises ImportError with a helpful message when the backend cannot
    be imported.
    """
    if name is None:
        env = os.environ.get("SENTENIAL_GUI_BACKEND")
        if env:
            name = env
        else:
            found = available_backends()
            if not found:
                raise ImportError(
                    "No GUI backends are available. Install one of: "
                    + ", ".join(_BACKEND_CANDIDATES)
                )
            name = found[0]

    # If the user passed a module object already, just return it.
    if not isinstance(name, str):
        return name

    # Prefer package submodule (apps.gui_desktop.<name>)
    submodule_path = f"{__name__}.{name}"
    try:
        spec = importlib.util.find_spec(submodule_path)
        if spec is not None:
            module = importlib.import_module(submodule_path)
            LOGGER.debug("Loaded GUI backend submodule %s", submodule_path)
            return module
    except Exception:
        # If a submodule exists but fails on import, surface the error.
        LOGGER.exception("Failed to import backend submodule %s", submodule_path)
        raise

    # Otherwise try to import it as a top-level module
    try:
        module = importlib.import_module(name)
        LOGGER.debug("Loaded GUI backend module %s", name)
        return module
    except Exception as exc:  # pragma: no cover - behavior depends on user's env
        raise ImportError(
            f"Could not import GUI backend '{name}'. Make sure it is installed. "
            "If you intended to use a bundled backend, ensure the submodule exists."
        ) from exc


def get_version() -> str:
    """Return the package version (best-effort)."""
    return __version__


# Backwards-friendly lazy attribute access. This lets consumers do:
#     from apps.gui_desktop import load_backend
# or access a default attribute dynamically.
def __getattr__(name: str):
    # Expose a convenience 'default_backend' attribute that returns the
    # module chosen by environment or discovery. Import only when requested.
    if name == "default_backend":
        try:
            return load_backend(None)
        except ImportError as exc:
            raise AttributeError("No default GUI backend available") from exc

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Small quality-of-life when used as a script for quick diagnostics.
if __name__ == "__main__":
    print(f"{__name__} version: {get_version()}")
    print("Available backends:", ", ".join(available_backends()))
