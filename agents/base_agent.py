# sentenial-x/agents/base_agent.py
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Abstract Base Class for all Sentenial-X agents.
    """

    def __init__(self, agent_id: str, meta: dict):
        self.agent_id = agent_id
        self.meta = meta
        self.status = "initialized"

    @abstractmethod
    def heartbeat(self):
        """
        Send periodic heartbeat to orchestrator
        """
        pass

    @abstractmethod
    def report_logs(self, logs: list):
        """
        Send logs or telemetry to orchestrator
        """
        pass

    @abstractmethod
    def execute_countermeasure(self, action: str):
        """
        Execute a countermeasure received from orchestrator
        """
        pass
