#!/usr/bin/env python3
import argparse
import logging
import sys
import asyncio
from datetime import datetime

from sentenial_core import (
    blind_spot_tracker,
    wormgpt_detector,
    cortex_analyzer,
    orchestrator
)
from memory import enqueue_command
from config import AGENT_ID

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ---- Command Handlers ---- #

async def run_wormgpt_detector(args):
    logging.info("Running WormGPT Detector / Counterstrike...")
    results = wormgpt_detector.analyze(prompt=args.prompt, temperature=args.temperature)
    print("\n[ WormGPT Detector Results ]")
    for line in results:
        print(f"- {line}")
    await enqueue_command(
        AGENT_ID,
        f"WormGPT Detector analyzed: {args.prompt[:60]}...",
        meta={"type": "wormgpt_detector", "results": results, "timestamp": datetime.utcnow().isoformat()}
    )

async def run_blind_spot_tracker(args):
    logging.info("Running Blind Spot Tracker...")
    findings = blind_spot_tracker.scan_blind_spots()
    print("\n[ Blind Spot Tracker Results ]")
    for issue in findings:
        print(f"- {issue}")
    await enqueue_command(
        AGENT_ID,
        "Blind Spot Tracker run",
        meta={"type": "blindspot_scan", "findings": findings, "timestamp": datetime.utcnow().isoformat()}
    )

async def run_cortex_analysis(args):
    logging.info("Running Cortex NLP Threat Analysis...")
    results = cortex_analyzer.analyze_logs(args.source, filter=args.filter)
    print("\n[ Cortex Analysis Results ]")
    for item in results:
        print(f"- {item}")
    await enqueue_command(
        AGENT_ID,
        f"Cortex analysis on {args.source}",
        meta={"type": "cortex_analysis", "results": results, "timestamp": datetime.utcnow().isoformat()}
    )

async def orchestrator_command(args):
    logging.info(f"Running orchestrator command: {args.action}")
    result = orchestrator.execute(args.action, params=args.params or {})
    print("\n[ Orchestrator Result ]")
    print(result)
    await enqueue_command(
        AGENT_ID,
        f"Orchestrator executed {args.action}",
        meta={"type": "orchestrator", "result": result, "timestamp": datetime.utcnow().isoformat()}
    )


# ---- CLI Entrypoint ---- #

def main():
    parser = argparse.ArgumentParser(description="Sentenial-X AI Defensive CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # WormGPT Detector
    worm_parser = subparsers.add_parser("wormgpt-detector", help="Simulate & counter WormGPT-style attacks")
    worm_parser.add_argument("-p", "--prompt", type=str, required=True, help="Adversarial input to analyze")
    worm_parser.add_argument("-t", "--temperature", type=float, default=0.7, help="Exploration randomness")
    worm_parser.set_defaults(func=lambda args: asyncio.run(run_wormgpt_detector(args)))

    # Blind Spot Tracker
    blind_parser = subparsers.add_parser("blindspots", help="Scan for detection blind spots")
    blind_parser.set_defaults(func=lambda args: asyncio.run(run_blind_spot_tracker(args)))

    # Cortex Analyzer
    cortex_parser = subparsers.add_parser("cortex", help="Run NLP-based threat analysis on logs")
    cortex_parser.add_argument("-s", "--source", type=str, required=True, help="Log source to analyze")
    cortex_parser.add_argument("-f", "--filter", type=str, default=None, help="Optional filter expression")
    cortex_parser.set_defaults(func=lambda args: asyncio.run(run_cortex_analysis(args)))

    # Orchestrator
    orch_parser = subparsers.add_parser("orchestrator", help="Execute orchestrator commands")
    orch_parser.add_argument("-a", "--action", type=str, required=True, help="Orchestrator action")
    orch_parser.add_argument("-p", "--params", type=dict, default=None, help="Optional parameters")
    orch_parser.set_defaults(func=lambda args: asyncio.run(orchestrator_command(args)))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
