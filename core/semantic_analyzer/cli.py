#!/usr/bin/env python3
"""
CLI for Semantic Analyzer
=========================

Provides a command-line interface for running the semantic analyzer
on telemetry logs or individual events.

Usage:
    python -m core.semantic_analyzer.cli analyze --file telemetry.json
    python -m core.semantic_analyzer.cli analyze --event '{"command": "powershell -ep bypass"}'
"""

import argparse
import json
import sys
from pathlib import Path

# Import analyzer functions
from core.semantic_analyzer.scoring import score_evasion


def analyze_event(event: dict, output_json: bool = False):
    """Analyze a single event and print/return results."""
    score = score_evasion(event)
    result = {"event": event, "stealth_score": score}

    if output_json:
        print(json.dumps(result, indent=2))
    else:
        print(f"[+] Command: {event.get('command', '<none>')}")
        print(f"    Stealth Score: {score:.2f}")

    return result


def analyze_file(file_path: Path, output_json: bool = False):
    """Analyze a JSON file containing events (list or single dict)."""
    if not file_path.exists():
        print(f"[!] File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    try:
        data = json.loads(file_path.read_text())
    except json.JSONDecodeError as e:
        print(f"[!] Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    results = []
    if isinstance(data, dict):
        results.append(analyze_event(data, output_json))
    elif isinstance(data, list):
        for event in data:
            results.append(analyze_event(event, output_json))
    else:
        print("[!] JSON must be object or array", file=sys.stderr)
        sys.exit(1)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Sentenial-X Semantic Analyzer CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze event(s)")
    group = analyze_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=Path, help="Path to JSON file with events")
    group.add_argument("--event", type=str, help="Raw JSON string of a single event")
    analyze_parser.add_argument(
        "--json", action="store_true", help="Output full JSON results"
    )

    args = parser.parse_args()

    if args.command == "analyze":
        if args.file:
            analyze_file(args.file, args.json)
        elif args.event:
            try:
                event = json.loads(args.event)
            except json.JSONDecodeError as e:
                print(f"[!] Invalid event JSON: {e}", file=sys.stderr)
                sys.exit(1)
            analyze_event(event, args.json)


if __name__ == "__main__":
    main()
