#!/usr/bin/env python3
"""
Sentenial X :: Enhanced App Entry Point

This script launches the Cortex-based analysis engine from the command line.
It supports:
  - Loading telemetry from a JSON file (with flexible timestamp parsing)
  - Auto-generating or accepting a session ID
  - Configurable verbosity and output formats (text or JSON)
  - Progress indication for large streams
  - Granular error handling with clear user feedback
"""

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dateutil.parser import parse as parse_timestamp
from tqdm import tqdm

from core.cortex.cortex_manager import CortexManager
from utils.logger import logger  # assumes a preconfigured logger instance

def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")
    logger.setLevel(level)

def load_telemetry(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load and normalize telemetry data from a JSON file.
    Returns a list of dicts with `timestamp` converted to datetime.
    Raises FileNotFoundError, json.JSONDecodeError, or ValueError for timestamp parse issues.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError("Telemetry JSON must be an array of entries.")

    normalized: List[Dict[str, Any]] = []
    for entry in tqdm(raw, desc="Normalizing telemetry", unit="event"):
        if not isinstance(entry, dict):
            logger.warning("Skipping non-dict entry: %r", entry)
            continue

        ts = entry.get("timestamp")
        if isinstance(ts, str):
            try:
                entry["timestamp"] = parse_timestamp(ts)
            except (ValueError, TypeError) as e:
                logger.error("Invalid timestamp format '%s': %s", ts, e)
                continue
        elif ts is None:
            logger.warning("Missing timestamp in entry, skipping: %r", entry)
            continue

        normalized.append(entry)
    return normalized

def output_results(
    result: Dict[str, Any],
    output_format: str = "text"
) -> None:
    """Print the analysis results in either 'text' or 'json' format."""
    if output_format == "json":
        # Ensure datetime serialization
        def _serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        print(json.dumps(result, default=_serializer, indent=2))
    else:
        # human-friendly summary
        print("\n=== Sentenial X Threat Analysis ===")
        print(f"Session ID       : {result['session_id']}")
        print(f"Semantic Intents : {len(result.get('semantic_intents', []))}")
        print(f"Threat Stories   : {len(result.get('threat_stories', []))}")
        print(f"Zero-Day Hits    : {len(result.get('zero_day_hits', []))}")
        print(f"Anomalies        : {len(result.get('anomalies', []))}")
        print("===================================\n")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sentenial X Cortex CLI (enhanced edition)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--telemetry",
        "-t",
        type=str,
        required=True,
        help="Path to telemetry JSON file"
    )
    parser.add_argument(
        "--session",
        "-s",
        type=str,
        default=str(uuid.uuid4()),
        help="Session ID (auto-generated if not provided)"
    )
    parser.add_argument(
        "--output-format",
        "-o",
        choices=["text", "json"],
        default="text",
        help="Output format"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging"
    )

    args = parser.parse_args()
    configure_logging(args.verbose)

    try:
        telemetry = load_telemetry(args.telemetry)
        if not telemetry:
            logger.error("No valid telemetry entries loaded. Exiting.")
            sys.exit(1)

    except FileNotFoundError as fnf:
        logger.error(fnf)
        sys.exit(2)
    except json.JSONDecodeError as je:
        logger.error("Failed to parse JSON: %s", je)
        sys.exit(3)
    except ValueError as ve:
        logger.error("Invalid telemetry format: %s", ve)
        sys.exit(4)

    logger.info("Loaded %d telemetry entries", len(telemetry))

    cortex = CortexManager()
    try:
        result = cortex.analyze(
            session_id=args.session,
            telemetry_stream=telemetry
        )
    except Exception as e:
        logger.exception("Analysis engine failed: %s", e)
        sys.exit(5)

    # Enrich result with summary counts if missing
    result.setdefault("session_id", args.session)
    output_results(result, output_format=args.output_format)


if __name__ == "__main__":
    main()
