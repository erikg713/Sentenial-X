# multiagent/coordinator.py
from typing import List
import logging
from agents.base_agent import BaseAgent

logger = logging.getLogger("SentenialX.MultiAgentCoordinator")


class MultiAgentCoordinator:
    """
    Simple lifecycle coordinator for multiple BaseAgent instances.
    Responsible for starting/stopping and gathering status.
    """

    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents

    def start_all(self):
        logger.info("Starting all agents")
        for a in self.agents:
            try:
                a.start()
            except Exception:
                logger.exception("Failed to start agent %s", a.agent_id)

    def stop_all(self):
        logger.info("Stopping all agents")
        for a in self.agents:
            try:
                a.stop()
            except Exception:
                logger.exception("Failed to stop agent %s", a.agent_id)

    def get_status(self):
        return {a.agent_id: a.get_status() for a in self.agents}
