"""
Sentenial-X AI
==============

Autonomous Cyber Defense Agents powered by Reinforcement Learning (RL)
and Machine Learning (ML). This package provides the core framework
for simulation, training, and real-time autonomous response.

Modules
-------
- agents: Base and specialized RL agents
- envs: Simulation and real-world telemetry environments
- policies: RL policies (DQN, PPO, A3C, etc.)
- orchestrator: Agent orchestration, coordination, and lifecycle
- utils: Logging, config management, monitoring tools
"""

__version__ = "0.1.0"
__author__ = "Sentenial-X Team"
__license__ = "Apache-2.0"

# Expose common entrypoints for easier imports
from sentenial_x.agents.base_agent import BaseAgent
from sentenial_x.agents.sentenial_agent import SentenialAgent

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "BaseAgent",
    "SentenialAgent",
]
