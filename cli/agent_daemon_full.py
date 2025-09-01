#!/usr/bin/env python3
"""
cli/agent_daemon_full.py

Full production-ready daemon for Sentenial-X:
- Manages all agents (AI, Retaliation, Endpoint)
- Async task scheduling
- Telemetry streaming
- Real-time logging to memory/SQLite
- Orchestrator integration
"""

import asyncio
import logging
import sys
import json
from datetime import datetime
from typing import Dict, Optional

# Import modules (replace with your actual imports)
from cli import wormgpt, cortex, alerts, orchestrator, telemetry, memory

# ------------------------------
# Logging setup
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agent_daemon_full")

# ------------------------------
# Agent registry
# ------------------------------
AGENTS: Dict[str, asyncio.Task] = {}

# ------------------------------
# Agent implementations
# ------------------------------
async def ai_agent():
    """AI threat detection agent using wormgpt"""
    while True:
        try:
            logger.info("[AI_AGENT] Running AI threat detection...")
            # Example: analyze prompts stored in memory
            prompts = memory.fetch_pending_prompts()
            for p in prompts:
                result = await wormgpt.analyze_prompt(p["prompt"])
                memory.log_result("ai_agent", p, result)
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"[AI_AGENT] Error: {e}")

async def retaliation_bot():
    """Retaliation/response agent"""
    while True:
        try:
            logger.info("[RETALIATION_BOT] Monitoring for active threats...")
            events = memory.fetch_alerts()
            for e in events:
                if e["severity"] == "high":
                    orchestrator.execute_block(e)
                    memory.log_result("retaliation_bot", e, {"action":"blocked"})
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"[RETALIATION_BOT] Error: {e}")

async def endpoint_agent():
    """Endpoint monitoring agent"""
    while True:
        try:
            logger.info("[ENDPOINT_AGENT] Scanning endpoints for anomalies...")
            telemetry_data = telemetry.fetch_endpoint_data()
            findings = cortex.analyze_endpoint(telemetry_data)
            for f in findings:
                memory.log_result("endpoint_agent", f, f)
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"[ENDPOINT_AGENT] Error: {e}")

AGENT_MAP = {
    "ai_agent": ai_agent,
    "retaliation_bot": retaliation_bot,
    "endpoint_agent": endpoint_agent,
}

# ------------------------------
# CLI control functions
# ------------------------------
def status(agent_name: Optional[str] = None):
    """Show status of all agents"""
    if agent_name:
        state = "running" if agent_name in AGENTS else "stopped"
        logger.info(f"{agent_name}: {state}")
    else:
        for name in AGENT_MAP:
            state = "running" if name in AGENTS else "stopped"
            logger.info(f"{name}: {state}")

async def start(agent_name: str):
    """Start an agent"""
    if agent_name not in AGENT_MAP:
        logger.error(f"Unknown agent: {agent_name}")
        return
    if agent_name in AGENTS:
        logger.warning(f"{agent_name} is already running")
        return
    task = asyncio.create_task(AGENT_MAP[agent_name](), name=agent_name)
    AGENTS[agent_name] = task
    logger.info(f"Started {agent_name}")

async def stop(agent_name: str):
    """Stop an agent gracefully"""
    task = AGENTS.get(agent_name)
    if not task:
        logger.warning(f"{agent_name} is not running")
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        logger.info(f"{agent_name} stopped")
    AGENTS.pop(agent_name, None)

async def restart(agent_name: str):
    """Restart an agent"""
    await stop(agent_name)
    await start(agent_name)

# ------------------------------
# Daemon CLI entrypoint
# ------------------------------
async def main():
    if len(sys.argv) < 3:
        logger.error("Usage: python -m cli.agent_daemon_full --start|--stop|--restart|--status <agent_name>")
        sys.exit(1)

    command, agent_name = sys.argv[1], sys.argv[2]

    if command == "--start":
        await start(agent_name)
    elif command == "--stop":
        await stop(agent_name)
    elif command == "--restart":
        await restart(agent_name)
    elif command == "--status":
        status(agent_name)
    else:
        logger.error(f"Unknown command: {command}")
        sys.exit(1)

    # Keep running any active agents
    running_tasks = [t for t in AGENTS.values() if not t.done()]
    if running_tasks:
        try:
            await asyncio.gather(*running_tasks)
        except asyncio.CancelledError:
            logger.info("Daemon shutting down...")

# ------------------------------
# Graceful shutdown handler
# ------------------------------
def run_daemon():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Received SIGINT. Cancelling all agents...")
        for task in list(AGENTS.values()):
            task.cancel()
        asyncio.run(asyncio.sleep(0.1))
        logger.info("Shutdown complete.")

# ------------------------------
# Entrypoint
# ------------------------------
if __name__ == "__main__":
    run_daemon()
