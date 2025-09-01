#!/usr/bin/env python3
"""
cli/cli.py

Main CLI entrypoint for Sentenial-X.
Supports:
- wormgpt-detector
- blindspots
- cortex
- orchestrator
- telemetry
- alert
- simulate

Usage:
    python cli.py <command> [options]
"""

import argparse
import asyncio
import logging
import sys

from cli import wormgpt, alerts, agent_daemon_full, memory, cortex, orchestrator, telemetry, simulate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SentenialCLI")


# ------------------------------
# Async command runner
# ------------------------------
async def run_command(args):
    """
    Dispatches the appropriate command handler based on CLI input.
    """
    try:
        if hasattr(args, "handler") and callable(args.handler):
            result = await args.handler(args)
            print(result)
        else:
            logger.error("No valid handler defined for command.")
    except Exception as e:
        logger.exception(f"Command execution failed: {e}")


# ------------------------------
# Subcommand handlers
# ------------------------------

# WormGPT
async def handle_wormgpt(args):
    return await wormgpt.run_detector(args.prompt, getattr(args, "temperature", 0.7))

# Blindspots
async def handle_blindspots(args):
    from cli.agent_daemon_full import scan_blindspots
    return await scan_blindspots()

# Cortex
async def handle_cortex(args):
    return await cortex.analyze(args.source, getattr(args, "filter", None))

# Orchestrator
async def handle_orchestrator(args):
    params = {}
    if getattr(args, "params", None):
        import json
        params = json.loads(args.params)
    return await orchestrator.execute(args.action, params)

# Telemetry
async def handle_telemetry(args):
    return await telemetry.stream(args.source, getattr(args, "filter", None))

# Alerts
async def handle_alert(args):
    import json
    metadata = json.loads(args.metadata) if getattr(args, "metadata", None) else None
    return await alerts.send_alert(args.type, getattr(args, "severity", "medium"), getattr(args, "message", None), metadata)

# Simulate
async def handle_simulate(args):
    return await simulate.run_scenario(args.scenario)


# ------------------------------
# CLI Parser Setup
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Sentenial-X CLI")
    subparsers = parser.add_subparsers(title="Commands", dest="command")

    # wormgpt-detector
    wp = subparsers.add_parser("wormgpt-detector", help="Detect adversarial AI prompts")
    wp.add_argument("-p", "--prompt", required=True, help="Prompt text to analyze")
    wp.add_argument("-t", "--temperature", type=float, default=0.7, help="Randomness for detection")
    wp.set_defaults(handler=handle_wormgpt)

    # blindspots
    bp = subparsers.add_parser("blindspots", help="Scan for detection blind spots")
    bp.set_defaults(handler=handle_blindspots)

    # cortex
    cp = subparsers.add_parser("cortex", help="Run NLP threat analysis")
    cp.add_argument("-s", "--source", required=True, help="Log file or event source")
    cp.add_argument("-f", "--filter", help="Optional filter expression")
    cp.set_defaults(handler=handle_cortex)

    # orchestrator
    op = subparsers.add_parser("orchestrator", help="Execute orchestrator actions")
    op.add_argument("-a", "--action", required=True, help="Orchestrator action name")
    op.add_argument("-p", "--params", help="JSON string of parameters")
    op.set_defaults(handler=handle_orchestrator)

    # telemetry
    tp = subparsers.add_parser("telemetry", help="Stream real-time telemetry")
    tp.add_argument("-s", "--source", required=True, help="Telemetry source")
    tp.add_argument("-f", "--filter", help="Optional filter expression")
    tp.set_defaults(handler=handle_telemetry)

    # alert
    ap = subparsers.add_parser("alert", help="Dispatch an alert")
    ap.add_argument("-t", "--type", required=True, help="Alert type")
    ap.add_argument("-s", "--severity", default="medium", help="Severity level")
    ap.add_argument("-m", "--message", help="Optional human-readable message")
    ap.add_argument("--metadata", help="Optional JSON metadata")
    ap.set_defaults(handler=handle_alert)

    # simulate
    sp = subparsers.add_parser("simulate", help="Run a threat simulation scenario")
    sp.add_argument("-sc", "--scenario", required=True, help="Threat scenario name")
    sp.set_defaults(handler=handle_simulate)

    # parse args
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Run async handler
    asyncio.run(run_command(args))


# ------------------------------
# Entry point
# ------------------------------
if __name__ == "__main__":
    main()
