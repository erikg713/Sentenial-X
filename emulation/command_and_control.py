"""
Sentenial-X Emulation: Command & Control (C2)
----------------------------------------------
Simulates a C2 server for threat emulation and red-team testing.
Designed for safe, contained cybersecurity exercises.
"""

import asyncio
from typing import Dict, Any, List
from sentenial_core.logger import logger

# Simulated agents connected to the C2 server
AGENTS: Dict[str, Dict[str, Any]] = {}


class Agent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.commands: List[str] = []
        self.responses: List[str] = []

    async def execute_command(self, command: str) -> str:
        """Simulate executing a command on the agent."""
        logger.info(f"[Agent {self.agent_id}] Executing command: {command}")
        self.commands.append(command)
        # Simulated delay to mimic real execution
        await asyncio.sleep(0.5)
        response = f"[Agent {self.agent_id}] Completed: {command}"
        self.responses.append(response)
        logger.info(response)
        return response


async def register_agent(agent_id: str) -> Agent:
    """Register a new agent with the C2 server."""
    if agent_id in AGENTS:
        logger.warning(f"Agent {agent_id} already registered.")
        return AGENTS[agent_id]["instance"]

    agent = Agent(agent_id)
    AGENTS[agent_id] = {"instance": agent}
    logger.info(f"Registered new agent: {agent_id}")
    return agent


async def send_command(agent_id: str, command: str) -> str:
    """Send a command to a specific agent."""
    agent_entry = AGENTS.get(agent_id)
    if not agent_entry:
        raise ValueError(f"Agent {agent_id} not registered.")
    agent = agent_entry["instance"]
    response = await agent.execute_command(command)
    return response


async def broadcast_command(command: str) -> Dict[str, str]:
    """Send a command to all connected agents."""
    if not AGENTS:
        logger.warning("No agents connected to broadcast command.")
        return {}
    results = {}
    for agent_id, entry in AGENTS.items():
        response = await entry["instance"].execute_command(command)
        results[agent_id] = response
    return results


# Example simulation
if __name__ == "__main__":
    async def main():
        # Register agents
        await register_agent("agent_01")
        await register_agent("agent_02")

        # Send commands
        response1 = await send_command("agent_01", "scan_network")
        response2 = await send_command("agent_02", "collect_logs")

        # Broadcast a command
        broadcast_results = await broadcast_command("simulate_malware_detection")

        print("Individual responses:")
        print(response1)
        print(response2)

        print("\nBroadcast results:")
        for aid, res in broadcast_results.items():
            print(f"{aid}: {res}")

    asyncio.run(main())
