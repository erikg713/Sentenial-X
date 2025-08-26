"""
Database Connector Module
=========================

Provides connections to relational (Postgres, SQLite) and NoSQL databases.
"""

import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor
from . import logger


def connect_sqlite(db_path: str = "sentenial.db"):
    """Connect to SQLite database."""
    logger.info(f"Connecting to SQLite DB at {db_path}")
    return sqlite3.connect(db_path)


def connect_postgres(
    host="localhost", port=5432, user="postgres", password="postgres", database="sentenial"
):
    """Connect to PostgreSQL database."""
    logger.info(f"Connecting to PostgreSQL at {host}:{port}/{database}")
    return psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=database,
        cursor_factory=RealDictCursor,
    )
