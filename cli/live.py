# cli/live.py

import asyncio
import logging
import sys
from datetime import datetime
from typing import Dict, Any

from agents.manager import AgentManager
from agents.endpoint_agent import EndpointAgent
from agents.sentenial_x_ai_bot import SentenialXAI
from agents.retaliation_bot import RetaliationBot
from agents.config import get_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger("SentenialX.LiveCLI")

# Initialize manager and agents
manager = AgentManager()
config = get_config()
manager.register_agent("endpoint_agent", EndpointAgent, config=config)
manager.register_agent("ai_agent", SentenialXAI, config=config)
manager.register_agent("retaliation_bot", RetaliationBot, config=config)


async def live_monitor():
    """Continuously display agent statuses and recent alerts."""
    while True:
        statuses = manager.list_agents()
        print("\n===== SENTENIAL-X AGENT STATUS =====")
        for name, running in statuses.items():
            print(f"{name:20}: {'RUNNING' if running else 'STOPPED'}")
        print("===================================\n")
        await asyncio.sleep(config.get("heartbeat_interval", 10))


async def telemetry_inspector():
    """Live telemetry inspection from AI agent's alerts."""
    ai_agent = manager.get_agent("ai_agent")
    if not ai_agent:
        logger.error("AI agent not found")
        return

    while True:
        try:
            # Fetch latest alert/event from AI agent (non-blocking)
            event = await ai_agent.alert_dispatcher.get_next_alert(timeout=2)
            if event:
                print(f"[{datetime.utcnow().isoformat()}] ALERT: {event}")
        except asyncio.TimeoutError:
            await asyncio.sleep(1)
        except Exception as e:
            logger.exception(f"Error in telemetry_inspector: {e}")


async def cli_input():
    """Handle live CLI input for commands."""
    print("\nLive CLI ready. Commands: start, stop, restart <agent>, exit, cm <enable/disable>")
    loop = asyncio.get_event_loop()

    while True:
        cmd = await loop.run_in_executor(None, input, "SENTENIAL-X> ")
        cmd_parts = cmd.strip().split()

        if not cmd_parts:
            continue
        action = cmd_parts[0].lower()

        if action == "start":
            await manager.start_all()
        elif action == "stop":
            await manager.stop_all()
        elif action == "restart" and len(cmd_parts) > 1:
            await manager.restart_agent(cmd_parts[1])
        elif action == "status":
            for name, running in manager.list_agents().items():
                print(f"{name:20}: {'RUNNING' if running else 'STOPPED'}")
        elif action == "cm" and len(cmd_parts) > 1:
            cm_agent = manager.get_agent("retaliation_bot")
            if cm_agent:
                if cmd_parts[1].lower() == "enable":
                    cm_agent.enable_countermeasures()
                elif cmd_parts[1].lower() == "disable":
                    cm_agent.disable_countermeasures()
        elif action == "exit":
            print("Exiting live CLI...")
            await manager.stop_all()
            break
        else:
            print("Unknown command. Commands: start, stop, restart <agent>, status, cm <enable/disable>, exit")


async def main():
    # Start all agents initially
    await manager.start_all()

    # Run all live tasks concurrently
    await asyncio.gather(
        live_monitor(),
        telemetry_inspector(),
        cli_input()
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[LiveCLI] Exiting...")
