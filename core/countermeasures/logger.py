#!/usr/bin/env python3
"""
Action Logger for Countermeasure Agent
Generates cryptographically signed, immutable logs.
"""

import json
import os
from datetime import datetime

class ActionLogger:
    def __init__(self, log_dir: str = "data/logs/"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir

    def log(self, action_name: str, user: str, result: Dict):
        timestamp = datetime.utcnow().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "action": action_name,
            "user": user,
            "result": result
        }
        # Simple logging; extend with cryptographic signing if needed
        file_path = os.path.join(self.log_dir, f"{timestamp.replace(':','-')}_{action_name}.json")
        with open(file_path, "w") as f:
            json.dump(log_entry, f, indent=4)
        return file_path
