"""
Sentenial-X AI Core Datastore
-----------------------------
Provides persistent storage for AI models, embeddings, inference logs,
and other AI Core artifacts. Supports local filesystem caching and
SQL databases for scalable storage.

Author: Sentenial-X Development Team
"""

import os
import sqlite3
import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional
from api.utils.logger import init_logger

logger = init_logger("ai_core.datastore")


class DataStore:
    """
    Simple thread-safe datastore for AI Core.
    By default uses SQLite, but can be extended for PostgreSQL or other backends.
    """

    _lock = threading.Lock()

    def __init__(self, db_path: Optional[str] = None):
        self.base_dir = Path(__file__).resolve().parent
        self.db_path = db_path or os.getenv("AI_DATASTORE_PATH", str(self.base_dir / "ai_core.db"))
        self._conn: Optional[sqlite3.Connection] = None
        self._connect()
        self._initialize_tables()

    def _connect(self):
        """Establish a thread-safe SQLite connection."""
        try:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            logger.info(f"Connected to AI datastore at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to datastore: {e}")
            raise

    def _initialize_tables(self):
        """Create default tables for model artifacts, embeddings, and logs."""
        try:
            with self._lock, self._conn:
                self._conn.execute("""
                    CREATE TABLE IF NOT EXISTS model_artifacts (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        type TEXT NOT NULL,
                        path TEXT NOT NULL,
                        metadata TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                self._conn.execute("""
                    CREATE TABLE IF NOT EXISTS inference_logs (
                        id TEXT PRIMARY KEY,
                        model_id TEXT NOT NULL,
                        input_data TEXT,
                        output_data TEXT,
                        confidence REAL,
                        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                self._conn.execute("""
                    CREATE TABLE IF NOT EXISTS embeddings (
                        id TEXT PRIMARY KEY,
                        model_id TEXT NOT NULL,
                        vector TEXT NOT NULL,
                        metadata TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            logger.info("Datastore tables initialized")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize tables: {e}")
            raise

    # ------------------------
    # CRUD Operations
    # ------------------------

    def insert_model_artifact(self, artifact_id: str, name: str, type_: str, path: str, metadata: Optional[Dict[str, Any]] = None):
        """Insert a model artifact record."""
        try:
            with self._lock, self._conn:
                self._conn.execute(
                    "INSERT OR REPLACE INTO model_artifacts (id, name, type, path, metadata) VALUES (?, ?, ?, ?, ?)",
                    (artifact_id, name, type_, path, json.dumps(metadata or {}))
                )
            logger.info(f"Model artifact inserted: {artifact_id} ({name})")
        except sqlite3.Error as e:
            logger.error(f"Failed to insert model artifact: {e}")

    def get_model_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a model artifact by ID."""
        try:
            with self._lock, self._conn:
                row = self._conn.execute("SELECT * FROM model_artifacts WHERE id = ?", (artifact_id,)).fetchone()
                if row:
                    return {k: json.loads(v) if k == "metadata" else v for k, v in dict(row).items()}
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch model artifact: {e}")
        return None

    def log_inference(self, log_id: str, model_id: str, input_data: Any, output_data: Any, confidence: float):
        """Log an inference call."""
        try:
            with self._lock, self._conn:
                self._conn.execute(
                    "INSERT INTO inference_logs (id, model_id, input_data, output_data, confidence) VALUES (?, ?, ?, ?, ?)",
                    (log_id, model_id, json.dumps(input_data), json.dumps(output_data), confidence)
                )
            logger.info(f"Inference logged: {log_id} (model {model_id})")
        except sqlite3.Error as e:
            logger.error(f"Failed to log inference: {e}")

    def insert_embedding(self, embedding_id: str, model_id: str, vector: list, metadata: Optional[Dict[str, Any]] = None):
        """Store an embedding vector."""
        try:
            with self._lock, self._conn:
                self._conn.execute(
                    "INSERT INTO embeddings (id, model_id, vector, metadata) VALUES (?, ?, ?, ?)",
                    (embedding_id, model_id, json.dumps(vector), json.dumps(metadata or {}))
                )
            logger.info(f"Embedding stored: {embedding_id} (model {model_id})")
        except sqlite3.Error as e:
            logger.error(f"Failed to insert embedding: {e}")

    # ------------------------
    # Utility
    # ------------------------
    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            logger.info("AI datastore connection closed")


# ------------------------
# Singleton Accessor
# ------------------------
_datastore_instance: Optional[DataStore] = None


def get_datastore() -> DataStore:
    """Return a singleton DataStore instance."""
    global _datastore_instance
    if _datastore_instance is None:
        _datastore_instance = DataStore()
    return _datastore_instance


# ------------------------
# Quick CLI Test
# ------------------------
if __name__ == "__main__":
    ds = get_datastore()
    ds.insert_model_artifact("mdl-001", "ThreatAnalyzer", "transformer", "/models/threatanalyzer.pt")
    artifact = ds.get_model_artifact("mdl-001")
    print("Retrieved Artifact:", artifact)

    ds.log_inference("log-001", "mdl-001", {"input": "test"}, {"output": "safe"}, 0.99)
    ds.insert_embedding("emb-001", "mdl-001", [0.12, 0.54, 0.23])
