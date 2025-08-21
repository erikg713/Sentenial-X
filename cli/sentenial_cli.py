#!/usr/bin/env python3
import argparse
import logging
import sys

from sentenial_core import blind_spot_tracker, wormgpt_detector
# from sentenial_core import cortex, analyzer, engine  # future expansions

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ---- Command Handlers ---- #

def run_wormgpt_detector(args):
    """
    Defensive module that simulates/adapts to WormGPT-style attacks
    and trains countermeasures.
    """
    logging.info("Running WormGPT Detector / Counterstrike...")
    results = wormgpt_detector.analyze(
        prompt=args.prompt,
        temperature=args.temperature
    )
    print("\n[ WormGPT Detector Results ]")
    for line in results:
        print(f"- {line}")


def run_blind_spot_tracker(args):
    logging.info("Running Blind Spot Tracker...")
    findings = blind_spot_tracker.scan_blind_spots()
    print("\n[ Blind Spot Tracker Results ]")
    for issue in findings:
        print(f"- {issue}")


# ---- Main Entrypoint ---- #

def main():
    parser = argparse.ArgumentParser(
        description="Sentenial-X AI Defensive CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # WormGPT Detector
    worm_parser = subparsers.add_parser(
        "wormgpt-detector",
        help="Simulate & counter WormGPT-style attacks"
    )
    worm_parser.add_argument(
        "-p", "--prompt", type=str, required=True,
        help="Adversarial input to analyze"
    )
    worm_parser.add_argument(
        "-t", "--temperature", type=float, default=0.7,
        help="Exploration randomness"
    )
    worm_parser.set_defaults(func=run_wormgpt_detector)

    # Blind Spot Tracker
    blind_parser = subparsers.add_parser(
        "blindspots", help="Scan for detection blind spots"
    )
    blind_parser.set_defaults(func=run_blind_spot_tracker)

    # TODO: cortex, analyzer, engine modules

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
