"""
agents

Core agent interfaces and base classes for the SentenialX A.I. system.
"""

from abc import ABC, abstractmethod
from typing import Any

__all__ = [
    "Agent",
]

class Agent(ABC):
    """
    Abstract base class for all agents in the SentenialX system.
    Agents are expected to implement the core operational method `process`.
    """

    @abstractmethod
    def process(self, *args: Any, **kwargs: Any) -> Any:
        """
        Core method for processing data or events.
        Must be implemented by all subclasses.
        """
        pass

# Clean up namespace
del ABC, abstractmethod, Any
