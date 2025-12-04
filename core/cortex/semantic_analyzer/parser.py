#!/usr/bin/env python3
"""
Sentenial-X :: Semantic Parser
==============================

Purpose:
    - Parse raw system events or logs into structured features
    - Normalize fields for semantic analysis
    - Preprocess textual and categorical data
    - Serve as input for SemanticAnalyzer, ZeroDayPredictor, and NLP models
"""

import json
import re
from typing import Dict, Any, List
from datetime import datetime


# ------------------------------------------------------------
# Event Parser
# ------------------------------------------------------------
class EventParser:
    """
    Parses raw event/log entries into structured dictionaries.
    """

    def __init__(self, normalize_timestamps: bool = True):
        self.normalize_timestamps = normalize_timestamps

    # --------------------------------------------------------
    # Parse a single raw event
    # --------------------------------------------------------
    def parse(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inputs:
            raw_event: dict with arbitrary keys (log_text, timestamp, user, syscalls, etc.)

        Outputs:
            parsed_event: normalized dict suitable for semantic analysis
        """
        parsed_event = {}

        # 1️⃣ Event ID
        parsed_event["event_id"] = raw_event.get("event_id") or self._generate_event_id(raw_event)

        # 2️⃣ Log Text
        parsed_event["log_text"] = str(raw_event.get("log_text", ""))

        # 3️⃣ Syscalls
        syscalls = raw_event.get("syscalls", [])
        if isinstance(syscalls, str):
            syscalls = [s.strip() for s in syscalls.split(",")]
        parsed_event["syscalls"] = syscalls

        # 4️⃣ User / Process
        parsed_event["user"] = raw_event.get("user", "unknown")
        parsed_event["process"] = raw_event.get("process", "unknown")

        # 5️⃣ Timestamp normalization
        ts = raw_event.get("timestamp")
        if ts and self.normalize_timestamps:
            parsed_event["timestamp"] = self._normalize_timestamp(ts)
        else:
            parsed_event["timestamp"] = ts or datetime.utcnow().isoformat()

        # 6️⃣ Additional fields (preserve)
        for k, v in raw_event.items():
            if k not in parsed_event:
                parsed_event[k] = v

        return parsed_event

    # --------------------------------------------------------
    # Batch parsing
    # --------------------------------------------------------
    def parse_batch(self, raw_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.parse(ev) for ev in raw_events]

    # --------------------------------------------------------
    # Generate deterministic event ID
    # --------------------------------------------------------
    def _generate_event_id(self, raw_event: Dict[str, Any]) -> str:
        log_snippet = raw_event.get("log_text", "")[:30]
        user = raw_event.get("user", "")
        ts = raw_event.get("timestamp", datetime.utcnow().isoformat())
        return f"EVT-{abs(hash(log_snippet + user + str(ts))) % (10 ** 8)}"

    # --------------------------------------------------------
    # Normalize timestamp to ISO8601
    # --------------------------------------------------------
    def _normalize_timestamp(self, ts: str) -> str:
        """
        Supports common timestamp formats, converts to ISO8601
        """
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%d-%m-%Y %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(ts, fmt)
                return dt.isoformat()
            except ValueError:
                continue
        # fallback
        return ts


# ------------------------------------------------------------
# CLI interface
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Semantic Event Parser CLI")
    parser.add_argument("--input", required=True, help="Input JSON file containing raw events")
    parser.add_argument("--output", required=False, help="Output JSON file path for parsed events")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        raw_events = json.load(f)

    parser_instance = EventParser()
    parsed_events = parser_instance.parse_batch(raw_events)

    output_json = json.dumps(parsed_events, indent=4)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"Parsed events saved to {args.output}")
    else:
        print(output_json)
