"""
Configuration loader and helpers.

This module centralizes how the application reads configuration from the
environment or .env files. It favors explicit conversions, helpful error
messages for missing/invalid configuration, and safe logging (secrets are
redacted).

Usage examples:
    DATABASE_URL = get("DATABASE_URL", default="sqlite:///sentenial.db")
    MAX_WORKERS = get_int("MAX_WORKERS", default=4)
    DEBUG = get_bool("DEBUG", default=False)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

from dotenv import find_dotenv, load_dotenv

from . import logger

# Attempt to load a .env file if present in the project tree (no override by default).
_env_path = find_dotenv()
if _env_path:
    load_dotenv(_env_path, override=False)
    logger.debug("Loaded .env from %s", _env_path)
else:
    logger.debug("No .env file found; relying on environment variables only.")


# Helpers ---------------------------------------------------------------------
_SENSITIVE_HINTS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "PWD", "DSN", "AWS", "GCP")


def _redact(key: str, value: Optional[str]) -> str:
    """Return a safe representation of a config value for logging."""
    if value is None:
        return "None"
    if any(hint in key.upper() for hint in _SENSITIVE_HINTS):
        # keep a tiny suffix so logs are somewhat useful without leaking secrets
        return f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "REDACTED"
    return value


def _cast_value(value: str, cast: Callable[[str], Any]) -> Any:
    """Apply casting function and surface a clear error if it fails."""
    try:
        return cast(value)
    except Exception as exc:  # keep broad to wrap underlying conversion errors
        raise ValueError(f"failed to cast value '{value}' using {cast}: {exc}") from exc


# Public API -----------------------------------------------------------------
def get(
    key: str,
    default: Optional[Any] = None,
    *,
    required: bool = False,
    cast: Optional[Callable[[str], Any]] = None,
    allow_empty: bool = True,
) -> Any:
    """
    Get an environment value.

    Parameters:
        key: environment variable name
        default: value to return when env var is not set
        required: if True, raise RuntimeError when no value is available
        cast: optional callable to convert the str to another type
        allow_empty: if False, treat empty string as missing

    Returns:
        the environment value (possibly cast) or default

    Raises:
        RuntimeError when required is True and value is missing/empty
        ValueError when cast fails
    """
    raw = os.getenv(key)
    if raw is None:
        value = default
        source = "default"
    else:
        if raw == "" and not allow_empty:
            value = default
            source = "default (empty not allowed)"
        else:
            value = raw
            source = "env"

    if required and value in (None, ""):
        raise RuntimeError(f"required configuration '{key}' is missing")

    if isinstance(value, str) and cast is not None:
        value = _cast_value(value, cast)

    logger.debug("Config: %s = %s (%s)", key, _redact(key, str(value) if value is not None else None), source)
    return value


def get_bool(key: str, default: Optional[bool] = None, *, required: bool = False) -> bool:
    """
    Read a boolean-ish env var.

    Accepts: 1/0, true/false, yes/no, y/n (case-insensitive).
    """
    def _to_bool(v: str) -> bool:
        v2 = v.strip().lower()
        if v2 in ("1", "true", "t", "yes", "y", "on"):
            return True
        if v2 in ("0", "false", "f", "no", "n", "off"):
            return False
        raise ValueError(f"unrecognized boolean value: {v!r}")

    result = get(key, default if default is not None else None, required=required, cast=_to_bool)
    # If default was provided as a bool and no env var set, return the default directly.
    if isinstance(result, bool):
        return result
    # If result is None and not required, return False for safety (unless default was None).
    if result is None:
        return False if default is None else default
    return bool(result)


def get_int(key: str, default: Optional[int] = None, *, required: bool = False) -> int:
    result = get(key, default if default is not None else None, required=required, cast=lambda v: int(v.strip()))
    if result is None:
        if default is None:
            raise RuntimeError(f"integer configuration '{key}' is required but missing")
        return default
    return int(result)


def get_float(key: str, default: Optional[float] = None, *, required: bool = False) -> float:
    result = get(key, default if default is not None else None, required=required, cast=lambda v: float(v.strip()))
    if result is None:
        if default is None:
            raise RuntimeError(f"float configuration '{key}' is required but missing")
        return default
    return float(result)


def get_list(key: str, default: Optional[Iterable[str]] = None, *, sep: str = ",", required: bool = False) -> List[str]:
    """
    Read a comma-separated list (or custom sep) and return a list of stripped strings.
    """
    def _to_list(v: str) -> List[str]:
        # filter out empty items that could appear from trailing commas
        return [item.strip() for item in v.split(sep) if item.strip()]

    result = get(key, None if default is None else sep.join(default), required=required, cast=_to_list)
    if result is None:
        return list(default) if default is not None else []
    return list(result)


def get_path(key: str, default: Optional[Path] = None, *, required: bool = False) -> Optional[Path]:
    """Return a Path object or default."""
    def _to_path(v: str) -> Path:
        return Path(v).expanduser().resolve()
    result = get(key, None if default is None else str(default), required=required, cast=_to_path)
    if result is None:
        return default
    return result


# Commonly used defaults (keep these lightweight and explicit)
DATABASE_URL: str = get("DATABASE_URL", "sqlite:///sentenial.db")
REDIS_URL: str = get("REDIS_URL", "redis://localhost:6379/0")
KAFKA_BROKER: str = get("KAFKA_BROKER", "localhost:9092")

__all__ = [
    "get",
    "get_bool",
    "get_int",
    "get_float",
    "get_list",
    "get_path",
    "DATABASE_URL",
    "REDIS_URL",
    "KAFKA_BROKER",
]
