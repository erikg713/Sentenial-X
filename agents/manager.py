# agents/manager.py

import asyncio
import logging
from typing import Dict, Type, Optional

from agents.base_agent import BaseAgent
from agents.config import AgentConfig

logger = logging.getLogger("SentenialX.AgentManager")


class AgentManager:
    """
    Central manager for orchestrating agents in Sentenial-X.
    Provides lifecycle control, secure message passing, and async coordination.
    """

    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._configs: Dict[str, AgentConfig] = {}

    def register_agent(self, name: str, agent_cls: Type[BaseAgent], config: Optional[AgentConfig] = None) -> None:
        """
        Register a new agent class with optional configuration.
        """
        if name in self._agents:
            logger.warning(f"Agent {name} is already registered. Skipping re-registration.")
            return

        cfg = config or AgentConfig(name=name)
        self._configs[name] = cfg
        self._agents[name] = agent_cls(cfg)
        logger.info(f"Registered agent '{name}' with configuration: {cfg}")

    async def start_agent(self, name: str) -> None:
        """
        Start a registered agent asynchronously.
        """
        agent = self._agents.get(name)
        if not agent:
            logger.error(f"Cannot start agent '{name}': not registered.")
            return

        if agent.is_running:
            logger.warning(f"Agent '{name}' is already running.")
            return

        logger.info(f"Starting agent '{name}'...")
        await agent.start()

    async def stop_agent(self, name: str) -> None:
        """
        Stop a running agent asynchronously.
        """
        agent = self._agents.get(name)
        if not agent:
            logger.error(f"Cannot stop agent '{name}': not registered.")
            return

        if not agent.is_running:
            logger.warning(f"Agent '{name}' is not running.")
            return

        logger.info(f"Stopping agent '{name}'...")
        await agent.stop()

    async def restart_agent(self, name: str) -> None:
        """
        Restart a running agent.
        """
        logger.info(f"Restarting agent '{name}'...")
        await self.stop_agent(name)
        await asyncio.sleep(1)
        await self.start_agent(name)

    async def start_all(self) -> None:
        """
        Start all registered agents.
        """
        logger.info("Starting all agents...")
        tasks = [self.start_agent(name) for name in self._agents]
        await asyncio.gather(*tasks)

    async def stop_all(self) -> None:
        """
        Stop all running agents.
        """
        logger.info("Stopping all agents...")
        tasks = [self.stop_agent(name) for name in self._agents]
        await asyncio.gather(*tasks)

    async def send_message(self, sender: str, receiver: str, message: dict) -> None:
        """
        Send a structured message from one agent to another.
        """
        if receiver not in self._agents:
            logger.error(f"Receiver agent '{receiver}' not found.")
            return

        agent = self._agents[receiver]
        try:
            await agent.receive_message(sender, message)
            logger.debug(f"Message sent from '{sender}' to '{receiver}': {message}")
        except Exception as e:
            logger.exception(f"Error delivering message to '{receiver}': {e}")

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """
        Retrieve an agent instance by name.
        """
        return self._agents.get(name)

    def list_agents(self) -> Dict[str, bool]:
        """
        Return a dictionary of agents and their running state.
        """
        return {name: agent.is_running for name, agent in self._agents.items()}


# Example usage (for manual testing)
if __name__ == "__main__":
    import asyncio
    from agents.endpoint_agent import EndpointAgent

    async def main():
        manager = AgentManager()
        manager.register_agent("endpoint", EndpointAgent)

        await manager.start_all()
        await asyncio.sleep(2)

        await manager.send_message("endpoint", "endpoint", {"action": "ping"})
        await asyncio.sleep(1)

        await manager.stop_all()

    asyncio.run(main())
