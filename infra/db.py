"""
Database connector utilities.

Goals / improvements made:
- Added environment-variable support for sensible defaults (DATABASE_URL, PG_*).
- Improved logging and error handling with retries and exponential backoff for Postgres.
- Added a lightweight connection pool manager for Postgres (psycopg2 SimpleConnectionPool).
- Provided context managers for getting/returning connections safely for both SQLite and Postgres.
- Set sqlite3.Row as the row factory so callers get dict-like access (more convenient & consistent
  with psycopg2 RealDictCursor).
- Type hints and cleaner public API with helpers to initialize/close the Postgres pool.
- Small input validation and docstrings for clarity.
"""

from __future__ import annotations

import os
import time
import sqlite3
from contextlib import contextmanager
from typing import Optional, Iterator

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool

from . import logger

__all__ = (
    "connect_sqlite",
    "sqlite_connection",
    "connect_postgres_once",
    "init_postgres_pool",
    "get_postgres_connection",
    "close_postgres_pool",
)

# Module-level pool (initialized via init_postgres_pool)
_PG_POOL: Optional[SimpleConnectionPool] = None


def _env_or(value: Optional[str], env_key: str, default: Optional[str] = None) -> Optional[str]:
    """Return `value` if truthy, otherwise check environment, otherwise default."""
    if value:
        return value
    return os.getenv(env_key, default)


def connect_sqlite(
    db_path: Optional[str] = None,
    timeout: float = 5.0,
    detect_types: int = sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
    check_same_thread: bool = False,
) -> sqlite3.Connection:
    """
    Create and return a new sqlite3.Connection.

    - Uses sqlite3.Row as row_factory for dict-like access.
    - By default, check_same_thread=False to be friendlier for multithreaded apps that
      hand off connections carefully (still recommend using connection-per-thread).
    """
    db_path = db_path or os.getenv("SQLITE_PATH", "sentenial.db")
    logger.info("Connecting to SQLite DB at %s", db_path)
    conn = sqlite3.connect(db_path, timeout=timeout, detect_types=detect_types, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def sqlite_connection(*, db_path: Optional[str] = None, **connect_kwargs) -> Iterator[sqlite3.Connection]:
    """
    Context manager that yields a sqlite3.Connection and ensures it's closed after use.

    Example:
        with sqlite_connection() as conn:
            cur = conn.execute("SELECT ...")
    """
    conn = connect_sqlite(db_path=db_path, **connect_kwargs)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception as exc:
            logger.exception("Error closing sqlite connection: %s", exc)


def connect_postgres_once(
    host: Optional[str] = None,
    port: Optional[int] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    database: Optional[str] = None,
    dsn: Optional[str] = None,
    cursor_factory=RealDictCursor,
    connect_timeout: int = 5,
    sslmode: Optional[str] = None,
    retries: int = 3,
    backoff_factor: float = 0.5,
) -> psycopg2.extensions.connection:
    """
    Create a single postgres connection with retries and exponential backoff.

    - If dsn is provided, it is used as-is (this can include environment-style DATABASE_URL).
    - Otherwise individual parameters or PG_* env vars are used.
    - Returns a psycopg2 connection (caller is responsible for closing it).
    """
    # Prefer DATABASE_URL / dsn if present
    dsn = dsn or os.getenv("DATABASE_URL") or dsn

    if dsn:
        conn_info = {"dsn": dsn}
    else:
        host = _env_or(host, "PGHOST", "localhost")
        port = int(_env_or(str(port) if port else None, "PGPORT", "5432") or 5432)
        user = _env_or(user, "PGUSER", "postgres")
        password = _env_or(password, "PGPASSWORD", "postgres")
        database = _env_or(database, "PGDATABASE", "sentenial")
        conn_info = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "dbname": database,
        }
        if sslmode:
            conn_info["sslmode"] = sslmode

    attempt = 0
    while True:
        try:
            logger.info("Connecting to PostgreSQL %s", conn_info if ds n else f"{host}:{port}/{database}")
        except Exception:
            logger.info("Connecting to PostgreSQL")
        try:
            # psycopg2.connect accepts a dsn kw or individual params
            conn = psycopg2.connect(cursor_factory=cursor_factory, connect_timeout=connect_timeout, **conn_info)
            # Ensure autocommit is disabled by default (caller can change)
            conn.autocommit = False
            return conn
        except Exception as exc:
            attempt += 1
            if attempt > retries:
                logger.exception("Failed to connect to Postgres after %d attempts: %s", attempt - 1, exc)
                raise
            sleep_time = backoff_factor * (2 ** (attempt - 1))
            logger.warning("Postgres connect attempt %d failed: %s; retrying in %.2fs", attempt, exc, sleep_time)
            time.sleep(sleep_time)


def init_postgres_pool(
    minconn: int = 1,
    maxconn: int = 10,
    host: Optional[str] = None,
    port: Optional[int] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    database: Optional[str] = None,
    dsn: Optional[str] = None,
    cursor_factory=RealDictCursor,
    connect_timeout: int = 5,
    sslmode: Optional[str] = None,
) -> SimpleConnectionPool:
    """
    Initialize a module-level Postgres connection pool and return it.

    Subsequent calls to get_postgres_connection() will draw from this pool.

    If a pool already exists, it will be returned as-is.
    """
    global _PG_POOL
    if _PG_POOL is not None:
        logger.debug("Postgres pool already initialized; returning existing pool")
        return _PG_POOL

    dsn = dsn or os.getenv("DATABASE_URL")
    if dsn:
        # When using a dsn/url we pass it as a single parameter to psycopg2
        conn_kwargs = {"dsn": dsn, "cursor_factory": cursor_factory, "connect_timeout": connect_timeout}
    else:
        host = _env_or(host, "PGHOST", "localhost")
        port = int(_env_or(str(port) if port else None, "PGPORT", "5432") or 5432)
        user = _env_or(user, "PGUSER", "postgres")
        password = _env_or(password, "PGPASSWORD", "postgres")
        database = _env_or(database, "PGDATABASE", "sentenial")
        conn_kwargs = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "dbname": database,
            "cursor_factory": cursor_factory,
            "connect_timeout": connect_timeout,
        }
        if sslmode:
            conn_kwargs["sslmode"] = sslmode

    # Create pool (this will attempt to open minconn connections immediately)
    logger.info("Initializing Postgres connection pool (min=%d max=%d)", minconn, maxconn)
    try:
        _PG_POOL = SimpleConnectionPool(minconn, maxconn, **conn_kwargs)
    except Exception as exc:
        logger.exception("Failed to initialize Postgres pool: %s", exc)
        raise

    return _PG_POOL


@contextmanager
def get_postgres_connection() -> Iterator[psycopg2.extensions.connection]:
    """
    Context manager that yields a connection from the module-level Postgres pool.

    Usage:
        with get_postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(...)

    If the pool isn't initialized, we initialize it using environment variables / defaults.
    """
    global _PG_POOL
    if _PG_POOL is None:
        logger.debug("Postgres pool not initialized; initializing with defaults from environment")
        init_postgres_pool()

    assert _PG_POOL is not None  # for type checkers
    conn = None
    try:
        conn = _PG_POOL.getconn()
        # Ensure the connection uses RealDictCursor by default when creating cursors, but we don't
        # enforce it here; callers can pass cursor_factory to cursor() if needed.
        yield conn
    except Exception as exc:
        logger.exception("Error while using a Postgres connection from pool: %s", exc)
        raise
    finally:
        if conn is not None:
            try:
                # Return the connection back to the pool for reuse
                _PG_POOL.putconn(conn)
            except Exception as exc:
                logger.exception("Failed to return Postgres connection to pool: %s", exc)


def close_postgres_pool() -> None:
    """Close and dispose of the module-level Postgres connection pool if present."""
    global _PG_POOL
    if _PG_POOL is None:
        return
    try:
        logger.info("Closing Postgres connection pool")
        _PG_POOL.closeall()
    except Exception as exc:
        logger.exception("Error closing Postgres pool: %s", exc)
    finally:
        _PG_POOL = None
