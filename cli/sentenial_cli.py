#!/usr/bin/env python3
"""
cli/sentenial_cli.py

Sentenial-X CLI - Production-ready, fully async operator interface.

Modules integrated:
- wormgpt: adversarial AI detection
- cortex: NLP threat analysis
- alerts: alert dispatcher
- orchestrator: central action executor
- telemetry: live telemetry streaming
- memory_adapter: logging and persistence

Usage:
./sentenial_cli.py <command> [options]
"""

import argparse
import asyncio
import json
import sys
from cli.wormgpt import WormGPT
from cli.cortex import Cortex
from cli.alerts import AlertDispatcher
from cli.orchestrator import get_orchestrator
from cli.telemetry import Telemetry
from cli.memory_adapter import get_adapter
from cli.logger import default_logger


# ------------------------------
# Async Command Handlers
# ------------------------------
async def handle_wormgpt(args):
    wg = WormGPT()
    res = await wg.analyze(prompt=args.prompt, temperature=args.temperature)
    await get_adapter().log_command("wormgpt-detector", {"prompt": args.prompt, "temperature": args.temperature}, res)
    print(json.dumps(res, indent=2))


async def handle_cortex(args):
    cortex = Cortex()
    res = await cortex.analyze(source=args.source, filter_expr=args.filter)
    await get_adapter().log_command("cortex", {"source": args.source, "filter": args.filter}, res)
    print(json.dumps(res, indent=2))


async def handle_alert(args):
    dispatcher = AlertDispatcher()
    res = await dispatcher.dispatch(alert_type=args.type, severity=args.severity)
    await get_adapter().log_command("alert", {"type": args.type, "severity": args.severity}, res)
    print(json.dumps(res, indent=2))


async def handle_orchestrator(args):
    orch = get_orchestrator()
    try:
        params = json.loads(args.params) if args.params else {}
    except json.JSONDecodeError:
        print(f"Invalid JSON for params: {args.params}")
        sys.exit(1)
    res = await orch.execute(action=args.action, params=params)
    print(json.dumps(res, indent=2))


async def handle_telemetry(args):
    telemetry = Telemetry()
    async for entry in telemetry.stream(source=args.source, filter_expr=args.filter):
        print(json.dumps(entry, indent=2))


async def handle_blindspots(args):
    # Example blindspot scan using Cortex + Orchestrator + Memory
    cortex = Cortex()
    orch = get_orchestrator()
    mem = get_adapter()
    res = await cortex.scan_blindspots()
    await mem.log_command("blindspots", {}, res)
    print(json.dumps(res, indent=2))


async def handle_simulate(args):
    # Placeholder simulation
    orch = get_orchestrator()
    mem = get_adapter()
    res = {"scenario": args.scenario, "status": "simulated"}
    await mem.log_command("simulate", {"scenario": args.scenario}, res)
    print(json.dumps(res, indent=2))


# ------------------------------
# CLI Argument Parser
# ------------------------------
def build_parser():
    parser = argparse.ArgumentParser(description="Sentenial-X CLI - async cybersecurity operator tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # wormgpt-detector
    wg = subparsers.add_parser("wormgpt-detector", help="Detect adversarial AI prompts")
    wg.add_argument("-p", "--prompt", type=str, required=True, help="Input prompt")
    wg.add_argument("-t", "--temperature", type=float, default=0.7, help="Exploration randomness")
    wg.set_defaults(func=handle_wormgpt)

    # cortex
    ctx = subparsers.add_parser("cortex", help="Run NLP threat analysis on logs/events")
    ctx.add_argument("-s", "--source", type=str, required=True, help="Log file or source")
    ctx.add_argument("-f", "--filter", type=str, default="", help="Optional filter expression")
    ctx.set_defaults(func=handle_cortex)

    # alert
    al = subparsers.add_parser("alert", help="Dispatch alerts")
    al.add_argument("-t", "--type", required=True, help="Alert type")
    al.add_argument("-s", "--severity", default="medium", help="Alert severity")
    al.set_defaults(func=handle_alert)

    # orchestrator
    orch = subparsers.add_parser("orchestrator", help="Execute orchestrator action")
    orch.add_argument("-a", "--action", required=True, help="Action name")
    orch.add_argument("-p", "--params", help="JSON parameters")
    orch.set_defaults(func=handle_orchestrator)

    # telemetry
    tm = subparsers.add_parser("telemetry", help="Stream live telemetry")
    tm.add_argument("-s", "--source", required=True, help="Telemetry source")
    tm.add_argument("-f", "--filter", default="", help="Filter expression")
    tm.set_defaults(func=handle_telemetry)

    # blindspots
    bs = subparsers.add_parser("blindspots", help="Scan for blind spots")
    bs.set_defaults(func=handle_blindspots)

    # simulate
    sim = subparsers.add_parser("simulate", help="Run threat simulation")
    sim.add_argument("-sc", "--scenario", required=True, help="Scenario name")
    sim.set_defaults(func=handle_simulate)

    return parser


# ------------------------------
# Main Async Runner
# ------------------------------
async def main():
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "func"):
        await args.func(args)
    else:
        parser.print_help()


# ------------------------------
# Entry Point
# ------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        default_logger.info("CLI interrupted by user.")
        print("\n[INFO] CLI interrupted by user. Exiting...")
