"""
Sentenial-X-A.I. Agents Package
==========================

This package contains modular agents used across the Sentenial-X
platform. Agents are responsible for specialized tasks such as
endpoint defense, reconnaissance, monitoring, and orchestration.

Each agent module should define a class that inherits from BaseAgent
and implements `start()` and `stop()` methods for consistent lifecycle
management.
"""

import importlib
import pkgutil
from typing import Dict, Type

from sentenial_x.core.base_agent import BaseAgent

__all__ = [
    "discover_agents",
    "get_agent_classes",
]

# Registry of agent classes
_AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {}


def discover_agents() -> None:
    """
    Dynamically discover all agent modules in this package
    and register their classes into the agent registry.
    """
    global _AGENT_REGISTRY
    package = __name__

    for _, module_name, is_pkg in pkgutil.iter_modules(__path__, f"{package}."):
        if is_pkg:
            continue

        try:
            module = importlib.import_module(module_name)

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseAgent)
                    and attr is not BaseAgent
                ):
                    _AGENT_REGISTRY[attr.__name__] = attr

        except Exception as e:
            # Prevent crash on bad import
            print(f"[Agents] Failed to load {module_name}: {e}")


def get_agent_classes() -> Dict[str, Type[BaseAgent]]:
    """
    Returns the dictionary of registered agent classes.
    Ensures discovery has run before returning.
    """
    if not _AGENT_REGISTRY:
        discover_agents()
    return _AGENT_REGISTRY


# Run discovery immediately so registry is available
discover_agents()
