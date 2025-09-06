# -*- coding: utf-8 -*-
"""
Sentenial-X CLI
---------------
Unified CLI for AI analysis, simulation, telemetry, and alerts.
"""

from __future__ import annotations
import argparse
import logging
from core.cortex import Cortex

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

cortex = Cortex()

def run_cortex(text: str, features: list[float] | None = None):
    report = cortex.full_analysis(text, features)
    print("=== Cortex Full Analysis Report ===")
    print(report)
    return report

def wormgpt_detector(text: str):
    report = cortex.analyze_text(text)
    print("=== WormGPT Detector Report ===")
    print(report)
    return report

def simulate(text: str, features: list[float] | None = None):
    report = cortex.full_analysis(text, features)
    print("=== Simulation Triggered ===")
    print(report)
    return report

def main():
    parser = argparse.ArgumentParser(description="Sentenial-X CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Cortex command
    parser_cortex = subparsers.add_parser("cortex")
    parser_cortex.add_argument("-t", "--text", required=True)
    parser_cortex.add_argument("-f", "--features", nargs="*", type=float)

    # WormGPT detector
    parser_worm = subparsers.add_parser("wormgpt-detector")
    parser_worm.add_argument("-t", "--text", required=True)

    # Simulate
    parser_sim = subparsers.add_parser("simulate")
    parser_sim.add_argument("-t", "--text", required=True)
    parser_sim.add_argument("-f", "--features", nargs="*", type=float)

    args = parser.parse_args()

    if args.command == "cortex":
        run_cortex(args.text, args.features)
    elif args.command == "wormgpt-detector":
        wormgpt_detector(args.text)
    elif args.command == "simulate":
        simulate(args.text, args.features)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
