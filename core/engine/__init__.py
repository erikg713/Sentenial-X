# -*- coding: utf-8 -*-
"""
core.engine package initializer
--------------------------------

This module provides:
- a stable __version__ lookup (best-effort via importlib.metadata),
- a module logger,
- a small Engine typing/protocol and a BaseEngine abstract helper for implementations,
- utility helpers for lazy-loading submodules and resolving engine implementations by name,
- PEP-562 based lazy-import support: accessing `core.engine.something` will try to import
  `core.engine.something` on demand (and cache the result).

Design goals:
- Minimal surface area so importing this package is cheap.
- Helpful error messages for common import/lookup mistakes.
- Typed and easy-to-read so it looks like it was written by a careful human engineer.
"""

from __future__ import annotations

import abc
import importlib
import importlib.metadata
import inspect
import logging
import pkgutil
import types
from contextlib import AbstractContextManager
from typing import Any, Optional, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Version handling - best-effort, non-fatal
# ---------------------------------------------------------------------------
def _get_package_version() -> str:
    """
    Try a list of likely distribution names and return the first found version.
    If none is found, return a sensible default.
    """
    candidates = (
        "Sentenial-X",
        "sentenial-x",
        "sentenial_x",
        "sentenialx",
        "core-engine",
        "core_engine",
    )
    for name in candidates:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception:
            # Any other metadata error shouldn't crash import; keep searching.
            continue
    return "0.0.0"


__version__ = _get_package_version()

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
# Provide a reasonable default handler only if nothing else configured by the app.
if not logger.handlers:
    # Keep default level WARNING so we don't spam applications that import us.
    logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Engine typing + base class
# ---------------------------------------------------------------------------
@runtime_checkable
class EngineProtocol(Protocol):
    """
    Lightweight protocol that engine implementations should satisfy.

    Minimal contract: implement `run` and optionally `start`/`stop` for lifecycle.
    Using a Protocol keeps this non-invasive while enabling static typing and
    runtime instance checks when desired.
    """

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the engine's primary behavior."""
        ...

    def start(self) -> None:
        """Optional: bring engine to a running state."""
        ...

    def stop(self) -> None:
        """Optional: gracefully stop the engine."""
        ...


class BaseEngine(AbstractContextManager, abc.ABC):
    """
    Small, opinionated base class for engine implementations.

    - Implements context manager behavior to call start/stop automatically.
    - Requires subclasses to implement `run`.
    - Offers no-op default implementations for start/stop for convenience.
    """

    def __enter__(self) -> "BaseEngine":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> Optional[bool]:
        # Stop should not suppress exceptions by default.
        try:
            self.stop()
        except Exception:
            logger.exception("Exception while stopping engine in context manager")
        return None

    def start(self) -> None:
        """Optional startup hook. Override if needed."""
        logger.debug("%s.start() called (default no-op)", self.__class__.__name__)

    def stop(self) -> None:
        """Optional shutdown hook. Override if needed."""
        logger.debug("%s.stop() called (default no-op)", self.__class__.__name__)

    @abc.abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Main entrypoint for the engine; must be implemented by subclasses."""
        raise NotImplementedError

# ---------------------------------------------------------------------------
# Lazy submodule loader + helpers
# ---------------------------------------------------------------------------
def load_submodule(name: str) -> types.ModuleType:
    """
    Import and return a submodule under core.engine.*

    Example:
        from core.engine import load_submodule
        mod = load_submodule("pipeline")  # imports core.engine.pipeline

    Raises:
        ImportError: if the submodule cannot be imported.
        ValueError: if name is malformed.
    """
    if not name or ".." in name or name.startswith(("/", "\\")):
        raise ValueError("Invalid submodule name: %r" % (name,))
    fq_name = f"{__name__}.{name}"
    try:
        module = importlib.import_module(fq_name)
        # Cache on package module namespace for subsequent attribute access
        setattr(importlib.import_module(__name__), name, module)
        return module
    except Exception as exc:  # keep exception breadth small for user-friendly message
        raise ImportError(f"Could not import {fq_name!r}: {exc!s}") from exc


def get_engine(name: str, *args: Any, **kwargs: Any) -> EngineProtocol:
    """
    Resolve and instantiate an engine implementation by submodule name.

    Resolution strategy (in order):
    - import core.engine.<name>; if the module defines `create_engine(...)`, call it.
    - if the module defines a class named `Engine`, instantiate it with provided args/kwargs.
    - if the module defines a module-level variable `engine` that implements the EngineProtocol, return it.
    - otherwise raise ImportError with guidance.

    This is a convenience helper used by applications that want pluggable engine backends.
    """
    module = load_submodule(name)

    # 1) factory function
    factory = getattr(module, "create_engine", None)
    if callable(factory):
        result = factory(*args, **kwargs)
        if isinstance(result, (EngineProtocol, BaseEngine)) or callable(getattr(result, "run", None)):
            return result  # type: ignore[return-value]
        # still return result if it looks like engine; otherwise keep checking

    # 2) Engine class
    cls = getattr(module, "Engine", None)
    if inspect.isclass(cls):
        inst = cls(*args, **kwargs)
        if isinstance(inst, EngineProtocol) or callable(getattr(inst, "run", None)):
            return inst  # type: ignore[return-value]

    # 3) engine instance
    inst = getattr(module, "engine", None)
    if inst is not None and (isinstance(inst, EngineProtocol) or callable(getattr(inst, "run", None))):
        return inst  # type: ignore[return-value]

    raise ImportError(
        f"Module {module.__name__!r} does not expose an engine implementation. "
        "Export one of:\n"
        "- a `create_engine(*args, **kwargs)` factory function,\n"
        "- an `Engine` class (instantiable), or\n"
        "- an `engine` instance with a `.run(...)` method."
    )

# PEP-562: module-level __getattr__ for lazy submodule imports
# When user does `import core.engine as eng` and then `eng.pipeline`,
# this will attempt to import core.engine.pipeline on demand.
def __getattr__(name: str) -> Any:  # pragma: no cover - small runtime helper
    try:
        # Fast path: existing attribute
        return globals()[name]
    except KeyError:
        pass

    # Try import as submodule of this package
    try:
        return load_submodule(name)
    except ImportError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc


def __dir__() -> list[str]:  # pragma: no cover - helper for auto-complete tools
    """
    Extend the package dir() with discovered submodules to improve auto-complete
    and introspection. We only list top-level modules found on the package __path__.
    """
    base = set(globals().keys())
    try:
        # pkgutil.iter_modules yields (finder, name, ispkg)
        names = {name for _, name, _ in pkgutil.iter_modules(__path__)}
        return sorted(base | names)
    except Exception:
        return sorted(base)


__all__ = [
    "__version__",
    "logger",
    "EngineProtocol",
    "BaseEngine",
    "load_submodule",
    "get_engine",
]    Try a list of likely distribution names and return the first found version.
    If none is found, return a sensible default.
    """
    candidates = (
        "Sentenial-X",
        "sentenial-x",
        "sentenial_x",
        "sentenialx",
        "core-engine",
        "core_engine",
    )
    for name in candidates:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception:
            # Any other metadata error shouldn't crash import; keep searching.
            continue
    return "0.0.0"


__version__ = _get_package_version()

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
# Provide a reasonable default handler only if nothing else configured by the app.
if not logger.handlers:
    # Keep default level WARNING so we don't spam applications that import us.
    logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Engine typing + base class
# ---------------------------------------------------------------------------
@runtime_checkable
class EngineProtocol(Protocol):
    """
    Lightweight protocol that engine implementations should satisfy.

    Minimal contract: implement `run` and optionally `start`/`stop` for lifecycle.
    Using a Protocol keeps this non-invasive while enabling static typing and
    runtime instance checks when desired.
    """

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the engine's primary behavior."""
        ...

    def start(self) -> None:
        """Optional: bring engine to a running state."""
        ...

    def stop(self) -> None:
        """Optional: gracefully stop the engine."""
        ...


class BaseEngine(AbstractContextManager, abc.ABC):
    """
    Small, opinionated base class for engine implementations.

    - Implements context manager behavior to call start/stop automatically.
    - Requires subclasses to implement `run`.
    - Offers no-op default implementations for start/stop for convenience.
    """

    def __enter__(self) -> "BaseEngine":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> Optional[bool]:
        # Stop should not suppress exceptions by default.
        try:
            self.stop()
        except Exception:
            logger.exception("Exception while stopping engine in context manager")
        return None

    def start(self) -> None:
        """Optional startup hook. Override if needed."""
        logger.debug("%s.start() called (default no-op)", self.__class__.__name__)

    def stop(self) -> None:
        """Optional shutdown hook. Override if needed."""
        logger.debug("%s.stop() called (default no-op)", self.__class__.__name__)

    @abc.abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Main entrypoint for the engine; must be implemented by subclasses."""
        raise NotImplementedError

# ---------------------------------------------------------------------------
# Lazy submodule loader + helpers
# ---------------------------------------------------------------------------
def load_submodule(name: str) -> types.ModuleType:
    """
    Import and return a submodule under core.engine.*

    Example:
        from core.engine import load_submodule
        mod = load_submodule("pipeline")  # imports core.engine.pipeline

    Raises:
        ImportError: if the submodule cannot be imported.
        ValueError: if name is malformed.
    """
    if not name or ".." in name or name.startswith(("/", "\\")):
        raise ValueError("Invalid submodule name: %r" % (name,))
    fq_name = f"{__name__}.{name}"
    try:
        module = importlib.import_module(fq_name)
        # Cache on package module namespace for subsequent attribute access
        setattr(importlib.import_module(__name__), name, module)
        return module
    except Exception as exc:  # keep exception breadth small for user-friendly message
        raise ImportError(f"Could not import {fq_name!r}: {exc!s}") from exc


def get_engine(name: str, *args: Any, **kwargs: Any) -> EngineProtocol:
    """
    Resolve and instantiate an engine implementation by submodule name.

    Resolution strategy (in order):
    - import core.engine.<name>; if the module defines `create_engine(...)`, call it.
    - if the module defines a class named `Engine`, instantiate it with provided args/kwargs.
    - if the module defines a module-level variable `engine` that implements the EngineProtocol, return it.
    - otherwise raise ImportError with guidance.

    This is a convenience helper used by applications that want pluggable engine backends.
    """
    module = load_submodule(name)

    # 1) factory function
    factory = getattr(module, "create_engine", None)
    if callable(factory):
        result = factory(*args, **kwargs)
        if isinstance(result, (EngineProtocol, BaseEngine)) or callable(getattr(result, "run", None)):
            return result  # type: ignore[return-value]
        # still return result if it looks like engine; otherwise keep checking

    # 2) Engine class
    cls = getattr(module, "Engine", None)
    if inspect.isclass(cls):
        inst = cls(*args, **kwargs)
        if isinstance(inst, EngineProtocol) or callable(getattr(inst, "run", None)):
            return inst  # type: ignore[return-value]

    # 3) engine instance
    inst = getattr(module, "engine", None)
    if inst is not None and (isinstance(inst, EngineProtocol) or callable(getattr(inst, "run", None))):
        return inst  # type: ignore[return-value]

    raise ImportError(
        f"Module {module.__name__!r} does not expose an engine implementation. "
        "Export one of:\n"
        "- a `create_engine(*args, **kwargs)` factory function,\n"
        "- an `Engine` class (instantiable), or\n"
        "- an `engine` instance with a `.run(...)` method."
    )

# PEP-562: module-level __getattr__ for lazy submodule imports
# When user does `import core.engine as eng` and then `eng.pipeline`,
# this will attempt to import core.engine.pipeline on demand.
def __getattr__(name: str) -> Any:  # pragma: no cover - small runtime helper
    try:
        # Fast path: existing attribute
        return globals()[name]
    except KeyError:
        pass

    # Try import as submodule of this package
    try:
        return load_submodule(name)
    except ImportError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc


def __dir__() -> list[str]:  # pragma: no cover - helper for auto-complete tools
    """
    Extend the package dir() with discovered submodules to improve auto-complete
    and introspection. We only list top-level modules found on the package __path__.
    """
    base = set(globals().keys())
    try:
        # pkgutil.iter_modules yields (finder, name, ispkg)
        names = {name for _, name, _ in pkgutil.iter_modules(__path__)}
        return sorted(base | names)
    except Exception:
        return sorted(base)


__all__ = [
    "__version__",
    "logger",
    "EngineProtocol",
    "BaseEngine",
    "load_submodule",
    "get_engine",
      ]
