"""core/ingestor.py - Optimized Data Ingestion for Sentenial-X A.I."""

import os
import json
import hashlib
import logging
from datetime import datetime
from typing import List, Dict, Optional, Set
from utils.logger import setup_logger

logger = setup_logger("ingestor")


class DataIngester:
    """
    Handles dynamic data ingestion and preprocessing from a specified directory.
    """

    def __init__(self, data_dir: str = "data/samples"):
        self.data_dir = data_dir
        self.processed_hashes: Set[str] = set()
        self.ingested_data: List[Dict] = []

    def _hash_file(self, filepath: str) -> Optional[str]:
        """
        Generates a SHA-256 hash of the specified file.
        Returns None if hashing fails.
        """
        sha256 = hashlib.sha256()
        try:
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as exc:
            logger.error(f"Hashing failed for {filepath}: {exc}")
            return None

    def ingest(self) -> List[Dict]:
        """
        Ingests files from the data directory, avoiding duplicates using file hashes.
        Returns a list of ingested data points.
        """
        logger.info(f"Starting ingestion from: {self.data_dir}")
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                path = os.path.join(root, file)
                file_hash = self._hash_file(path)
                if not file_hash or file_hash in self.processed_hashes:
                    continue

                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    data_point = {
                        "filename": file,
                        "path": path,
                        "hash": file_hash,
                        "content": content,
                        "timestamp": datetime.utcnow().isoformat()
                    }

                    self.ingested_data.append(data_point)
                    self.processed_hashes.add(file_hash)
                    logger.debug(f"Ingested: {file} [{file_hash}]")

                except Exception as exc:
                    logger.error(f"Failed to ingest {path}: {exc}")

        logger.info(f"Ingestion complete: {len(self.ingested_data)} new items.")
        return self.ingested_data

    def export_json(self, output_path: str = "logs/ingested_data.json") -> None:
        """
        Exports ingested data to a JSON file.
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as out:
                json.dump(self.ingested_data, out, indent=2, ensure_ascii=False)
            logger.info(f"Exported ingested data to {output_path}")
        except Exception as exc:
            logger.error(f"Failed to export ingested data: {exc}")


if __name__ == "__main__":
    ingester = DataIngester()
    ingester.ingest()
    ingester.export_json()
