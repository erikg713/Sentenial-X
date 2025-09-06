# -*- coding: utf-8 -*-
"""
Database Migration Script for Sentenial-X
----------------------------------------

- Handles schema migrations for PostgreSQL
- Applies migrations sequentially
- Maintains a migration history table
- Supports dry-run and verbose logging
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SentenialX.DBMigration")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = int(os.environ.get("DB_PORT", 5432))
DB_NAME = os.environ.get("DB_NAME", "sentenialx")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "password")

MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations"

# ---------------------------------------------------------------------------
# Connect to DB
# ---------------------------------------------------------------------------
def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )
        logger.info("Connected to database %s@%s:%s", DB_USER, DB_HOST, DB_PORT)
        return conn
    except Exception as e:
        logger.exception("Failed to connect to database: %s", e)
        sys.exit(1)

# ---------------------------------------------------------------------------
# Ensure migrations table
# ---------------------------------------------------------------------------
def ensure_migrations_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id SERIAL PRIMARY KEY,
                migration_file TEXT NOT NULL UNIQUE,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
    logger.info("Migrations table ensured.")

# ---------------------------------------------------------------------------
# Discover migration scripts
# ---------------------------------------------------------------------------
def discover_migrations(migrations_dir: Path) -> List[Path]:
    if not migrations_dir.exists():
        logger.warning("Migrations directory not found: %s", migrations_dir)
        return []

    migrations = sorted(migrations_dir.glob("*.sql"))
    logger.info("Discovered %d migration(s) in %s", len(migrations), migrations_dir)
    return migrations

# ---------------------------------------------------------------------------
# Apply migrations
# ---------------------------------------------------------------------------
def apply_migrations(conn, migrations: List[Path], dry_run: bool = False):
    applied = []
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        for migration in migrations:
            # Check if migration already applied
            cur.execute("SELECT 1 FROM schema_migrations WHERE migration_file = %s", (migration.name,))
            if cur.fetchone():
                logger.info("Skipping already applied migration: %s", migration.name)
                continue

            # Read SQL
            sql = migration.read_text()
            logger.info("Applying migration: %s", migration.name)
            if dry_run:
                logger.info("[Dry Run] SQL:\n%s", sql)
                continue

            try:
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO schema_migrations (migration_file) VALUES (%s)", (migration.name,)
                )
                conn.commit()
                applied.append(migration.name)
                logger.info("Migration applied successfully: %s", migration.name)
            except Exception as e:
                conn.rollback()
                logger.exception("Failed to apply migration %s: %s", migration.name, e)
                break
    return applied

# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def main(dry_run: bool = False):
    conn = get_db_connection()
    ensure_migrations_table(conn)
    migrations = discover_migrations(MIGRATIONS_DIR)
    applied = apply_migrations(conn, migrations, dry_run=dry_run)
    logger.info("Total migrations applied: %d", len(applied))
    if applied:
        print("Applied migrations:", applied)
    else:
        print("No new migrations applied.")
    conn.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply database migrations for Sentenial-X.")
    parser.add_argument("--dry-run", action="store_true", help="Print SQL but do not apply")
    args = parser.parse_args()

    main(dry_run=args.dry_run)
