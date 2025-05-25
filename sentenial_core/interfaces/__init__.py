"""
sentenial_core.interfaces

Defines core interface contracts for the SentenialX system.
"""

from abc import ABC, abstractmethod

__all__ = [
    # List public interface classes here as you add them, e.g.:
    # "DataProvider",
    # "EventListener",
]

# Example interface template for future use.
class InterfaceTemplate(ABC):
    """
    Abstract base class template for SentenialX interfaces.
    """

    @abstractmethod
    def process(self, *args, **kwargs):
        """
        Process data or events. Should be implemented by subclasses.
        """
        pass

# Clean up namespace
del ABC, abstractmethod
