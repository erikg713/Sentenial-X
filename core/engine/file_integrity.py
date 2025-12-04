#!/usr/bin/env python3
"""
Sentenial-X :: File Integrity Monitoring (FIM)
==============================================

Purpose:
    Monitor critical files or directories for unauthorized changes.
    Supports hash-based detection, logging, and optional alert dispatching.
"""

import os
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional

from core.engine.alert_dispatcher import AlertDispatcher


# ------------------------------------------------------------
# File Integrity Monitor
# ------------------------------------------------------------
class FileIntegrityMonitor:
    def __init__(self, paths: List[str], alert_dispatcher: Optional[AlertDispatcher] = None, log_dir: str = "data/logs/file_integrity/"):
        """
        paths: list of files or directories to monitor
        alert_dispatcher: optional AlertDispatcher to notify on changes
        log_dir: directory to store integrity logs
        """
        self.paths = paths
        self.alert_dispatcher = alert_dispatcher
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.state_file = os.path.join(log_dir, "fim_state.json")
        self.state = self._load_state()

    # --------------------------------------------------------
    # Compute SHA256 hash of a file
    # --------------------------------------------------------
    def _hash_file(self, file_path: str) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    # --------------------------------------------------------
    # Scan all monitored paths
    # --------------------------------------------------------
    def scan(self) -> Dict[str, Dict[str, str]]:
        """
        Returns a dict: {file_path: {"hash": ..., "status": "unchanged|modified|new|deleted"}}
        """
        results = {}
        current_state = {}

        for path in self.paths:
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for fname in files:
                        fpath = os.path.join(root, fname)
                        h = self._hash_file(fpath)
                        current_state[fpath] = h
            elif os.path.isfile(path):
                h = self._hash_file(path)
                current_state[path] = h
            else:
                continue  # skip non-existent

        # Compare with previous state
        for fpath, h in current_state.items():
            prev_hash = self.state.get(fpath)
            if prev_hash is None:
                status = "new"
            elif prev_hash != h:
                status = "modified"
            else:
                status = "unchanged"
            results[fpath] = {"hash": h, "status": status}
            # Alert if changed
            if status in ["modified", "new"] and self.alert_dispatcher:
                self.alert_dispatcher.dispatch_alert({
                    "title": f"File Integrity Alert: {os.path.basename(fpath)}",
                    "message": f"File {fpath} is {status}.",
                    "source": "FileIntegrityMonitor",
                    "severity": "high",
                    "context": {"file_path": fpath, "status": status}
                }, priority="high")

        # Detect deleted files
        for fpath in self.state.keys():
            if fpath not in current_state:
                results[fpath] = {"hash": None, "status": "deleted"}
                if self.alert_dispatcher:
                    self.alert_dispatcher.dispatch_alert({
                        "title": f"File Integrity Alert: {os.path.basename(fpath)}",
                        "message": f"File {fpath} has been deleted.",
                        "source": "FileIntegrityMonitor",
                        "severity": "high",
                        "context": {"file_path": fpath, "status": "deleted"}
                    }, priority="high")

        # Update state
        self.state = current_state
        self._save_state()
        return results

    # --------------------------------------------------------
    # Load saved file state
    # --------------------------------------------------------
    def _load_state(self) -> Dict[str, str]:
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                return json.load(f)
        return {}

    # --------------------------------------------------------
    # Save current file state
    # --------------------------------------------------------
    def _save_state(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=4)

# ------------------------------------------------------------
# CLI interface
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sentenial-X File Integrity Monitor CLI")
    parser.add_argument("--paths", required=True, nargs="+", help="Files or directories to monitor")
    args = parser.parse_args()

    fim = FileIntegrityMonitor(paths=args.paths)
    results = fim.scan()
    print(json.dumps(results, indent=4))
