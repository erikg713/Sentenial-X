""" core/ingester.py Optimized for SentenialX A.I - handles dynamic data stream ingestion and preprocessing. """

import os import json import hashlib import logging from datetime import datetime from utils.logger import setup_logger

logger = setup_logger("ingester")

class DataIngester: def init(self, data_dir="data/samples"): self.data_dir = data_dir self.processed_hashes = set() self.ingested_data = []

def _hash_file(self, filepath):
    try:
        with open(filepath, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        logger.error(f"Hashing failed for {filepath}: {e}")
        return None

def ingest(self):
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

            except Exception as e:
                logger.error(f"Failed to ingest {path}: {e}")

    logger.info(f"Ingestion complete: {len(self.ingested_data)} new items.")
    return self.ingested_data

def export_json(self, output_path="logs/ingested_data.json"):
    try:
        with open(output_path, 'w') as out:
            json.dump(self.ingested_data, out, indent=2)
        logger.info(f"Exported ingested data to {output_path}")
    except Exception as e:
        logger.error(f"Failed to export ingested data: {e}")

if name == "main": ingester = DataIngester() ingester.ingest() ingester.export_json()

