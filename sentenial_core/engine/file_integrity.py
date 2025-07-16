# engine/file_integrity.py

import os
import hashlib
import json
from pathlib import Path

class FileIntegrityMonitor:
    def __init__(self, paths):
        self.paths = paths
        self.snapshot_file = "file_integrity_baseline.json"
        self.baseline = self._load_snapshot()

    def _hash_file(self, path):
        try:
            with open(path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return None

    def _load_snapshot(self):
        if Path(self.snapshot_file).exists():
            with open(self.snapshot_file, "r") as f:
                return json.load(f)
        return {}

    def _save_snapshot(self):
        with open(self.snapshot_file, "w") as f:
            json.dump(self.baseline, f, indent=2)

    def scan(self):
        alerts = []
        for base_path in self.paths:
            for root, _, files in os.walk(base_path):
                for name in files:
                    full_path = os.path.join(root, name)
                    h = self._hash_file(full_path)
                    if not h:
                        continue
                    prev_hash = self.baseline.get(full_path)
                    if prev_hash and prev_hash != h:
                        alerts.append(f"MODIFIED: {full_path}")
                    elif not prev_hash:
                        alerts.append(f"NEW FILE: {full_path}")
                    self.baseline[full_path] = h
        self._save_snapshot()
        return alerts