import abc
from typing import Dict, Any

class BaseScanner(abc.ABC):
    """
    Abstract base class for all custom scanners.
    """

    @abc.abstractmethod
    def scan(self, target: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the scanner on the given target.

        Args:
            target (str): The target to scan (file, host, system)
            options (Dict[str, Any], optional): Scanner-specific options

        Returns:
            Dict[str, Any]: Scan results
        """
        pass

    @abc.abstractmethod
    def describe(self) -> str:
        """Return a description of the scanner."""
        pass
