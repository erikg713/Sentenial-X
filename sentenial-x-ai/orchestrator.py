"""
Sentenial-X AI Orchestrator
===========================

Manages multiple autonomous agents:
- Lifecycle: start, stop, restart
- Logging and monitoring
- Configuration management
- Multi-agent coordination
"""

import logging
from typing import Dict, Optional
from threading import Lock
from sentenial_x.config import Config
from sentenial_x.agents.base_agent import BaseAgent
from sentenial_x.agents.sentenial_agent import SentenialAgent

logger = logging.getLogger("SentenialX.Orchestrator")
logger.setLevel(Config.LOG_LEVEL)


class Orchestrator:
    """
    Orchestrator for managing multiple Sentenial-X agents.
    """

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.lock = Lock()
        logger.info("Sentenial-X Orchestrator initialized.")

    def register_agent(self, agent_id: str, agent: BaseAgent):
        """Register a new agent with a unique ID."""
        with self.lock:
            if agent_id in self.agents:
                logger.warning(f"Agent {agent_id} is already registered.")
                return
            self.agents[agent_id] = agent
            logger.info(f"Agent {agent_id} registered.")

    def start_agent(self, agent_id: str):
        """Start a registered agent."""
        agent = self.agents.get(agent_id)
        if not agent:
            logger.error(f"Agent {agent_id} not found.")
            return
        logger.info(f"Starting agent {agent_id}...")
        agent.start()

    def stop_agent(self, agent_id: str):
        """Stop a registered agent."""
        agent = self.agents.get(agent_id)
        if not agent:
            logger.error(f"Agent {agent_id} not found.")
            return
        logger.info(f"Stopping agent {agent_id}...")
        agent.stop()

    def get_agent_status(self, agent_id: str) -> Optional[dict]:
        """Return the status of a registered agent."""
        agent = self.agents.get(agent_id)
        if not agent:
            logger.error(f"Agent {agent_id} not found.")
            return None
        return agent.get_status()

    def stop_all(self):
        """Stop all registered agents."""
        logger.info("Stopping all agents...")
        with self.lock:
            for agent_id, agent in self.agents.items():
                logger.info(f"Stopping agent {agent_id}...")
                agent.stop()

    def list_agents(self):
        """Return a list of all registered agents and their status."""
        with self.lock:
            return {agent_id: agent.get_status() for agent_id, agent in self.agents.items()}


# Example usage:
if __name__ == "__main__":
    import time
    from cyberbattle.simulation import env

    # Initialize orchestrator
    orchestrator = Orchestrator()

    # Create a test agent
    environment = env.CyberBattleEnv()
    agent = SentenialAgent(environment)

    # Register agent
    orchestrator.register_agent("agent001", agent.agent)

    # Start agent
    orchestrator.start_agent("agent001")

    # Monitor for 10 seconds
    try:
        for _ in range(10):
            status = orchestrator.get_agent_status("agent001")
            logger.info(f"Agent status: {status}")
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop all agents
        orchestrator.stop_all()
        logger.info("Orchestrator shutdown complete.")
