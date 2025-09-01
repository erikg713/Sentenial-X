# cli/agent.py

import asyncio
import logging
import sys
from typing import Optional

from agents.manager import AgentManager
from agents.endpoint_agent import EndpointAgent
from agents.sentenial_x_ai_bot import SentenialXAI
from agents.retaliation_bot import RetaliationBot
from agents.config import get_config

# ==============================
# Logging
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger("SentenialX.AgentCLI")

# ==============================
# Initialize Agent Manager
# ==============================
manager = AgentManager()
config = get_config()
manager.register_agent("endpoint_agent", EndpointAgent, config=config)
manager.register_agent("ai_agent", SentenialXAI, config=config)
manager.register_agent("retaliation_bot", RetaliationBot, config=config)


async def start_agent(agent_name: str):
    agent = manager.get_agent(agent_name)
    if not agent:
        logger.error(f"Agent '{agent_name}' not found.")
        return
    await manager.start_agent(agent_name)
    logger.info(f"Agent '{agent_name}' started.")


async def stop_agent(agent_name: str):
    agent = manager.get_agent(agent_name)
    if not agent:
        logger.error(f"Agent '{agent_name}' not found.")
        return
    await manager.stop_agent(agent_name)
    logger.info(f"Agent '{agent_name}' stopped.")


async def restart_agent(agent_name: str):
    agent = manager.get_agent(agent_name)
    if not agent:
        logger.error(f"Agent '{agent_name}' not found.")
        return
    await manager.restart_agent(agent_name)
    logger.info(f"Agent '{agent_name}' restarted.")


def status(agent_name: Optional[str] = None):
    if agent_name:
        agent = manager.get_agent(agent_name)
        if agent:
            print(f"{agent_name}: {'RUNNING' if agent.is_running else 'STOPPED'}")
        else:
            print(f"Agent '{agent_name}' not found.")
    else:
        all_status = manager.list_agents()
        for name, running in all_status.items():
            print(f"{name:20}: {'RUNNING' if running else 'STOPPED'}")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sentenial-X Agent CLI")
    parser.add_argument("--start", type=str, help="Start a specific agent")
    parser.add_argument("--stop", type=str, help="Stop a specific agent")
    parser.add_argument("--restart", type=str, help="Restart a specific agent")
    parser.add_argument("--status", type=str, nargs="?", const=None, help="Show status of a specific agent or all")

    args = parser.parse_args()

    if args.start:
        await start_agent(args.start)
    elif args.stop:
        await stop_agent(args.stop)
    elif args.restart:
        await restart_agent(args.restart)
    elif args.status is not None:
        status(args.status)
    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[AgentCLI] Exiting...")
