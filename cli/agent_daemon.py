#!/usr/bin/env python3
"""
cli/agent_daemon.py

Production-ready daemon for managing Sentenial-X agents:
- AI Agent
- Retaliation Bot
- Endpoint Agent

Features:
- Async task management
- Logging
- Start/Stop/Restart/Status commands
- Graceful shutdown
"""

import asyncio
import logging
import sys
from datetime import datetime
from typing import Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agent_daemon")

# Registry for running agents
AGENTS: Dict[str, asyncio.Task] = {}

# ------------------------------
# Agent implementations (replace with real logic)
# ------------------------------
async def ai_agent():
    while True:
        logger.info("[AI_AGENT] Processing AI threat detection...")
        await asyncio.sleep(5)

async def retaliation_bot():
    while True:
        logger.info("[RETALIATION_BOT] Monitoring for retaliation events...")
        await asyncio.sleep(5)

async def endpoint_agent():
    while True:
        logger.info("[ENDPOINT_AGENT] Monitoring endpoints for anomalies...")
        await asyncio.sleep(5)

AGENT_MAP = {
    "ai_agent": ai_agent,
    "retaliation_bot": retaliation_bot,
    "endpoint_agent": endpoint_agent,
}

# ------------------------------
# Core agent control functions
# ------------------------------
def status(agent_name: Optional[str] = None):
    """Print status of agents"""
    if agent_name:
        state = "running" if agent_name in AGENTS else "stopped"
        logger.info(f"{agent_name}: {state}")
    else:
        for name in AGENT_MAP:
            state = "running" if name in AGENTS else "stopped"
            logger.info(f"{name}: {state}")

async def start(agent_name: str):
    """Start an agent asynchronously"""
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
    """Stop a running agent gracefully"""
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
# CLI entrypoint
# ------------------------------
async def main():
    if len(sys.argv) < 3:
        logger.error("Usage: python -m cli.agent --start|--stop|--restart|--status <agent_name>")
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

    # Keep running any started agents
    running_tasks = [t for t in AGENTS.values() if not t.done()]
    if running_tasks:
        try:
            await asyncio.gather(*running_tasks)
        except asyncio.CancelledError:
            logger.info("Daemon shutting down...")

# ------------------------------
# Graceful shutdown handling
# ------------------------------
def run_daemon():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Received SIGINT. Cancelling running agents...")
        for task in list(AGENTS.values()):
            task.cancel()
        asyncio.run(asyncio.sleep(0.1))
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    run_daemon()
