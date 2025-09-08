# agents/base_agent.py

import abc
import logging
import threading
import time
from typing import Dict, Any, Optional


class BaseAgent(abc.ABC):
    """
    BaseAgent serves as the foundation for all Sentenial-X agents.
    It enforces lifecycle methods and provides logging, configuration,
    and thread management utilities.
    """

    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the agent with an ID and optional configuration.

        :param agent_id: Unique identifier for the agent.
        :param config: Dictionary of configuration options.
        """
        self.agent_id = agent_id
        self.config = config or {}
        self.running = False
        self._thread: Optional[threading.Thread] = None

        self.logger = logging.getLogger(f"SentenialX.Agent.{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)

    @abc.abstractmethod
    def setup(self):
        """
        Setup phase for initializing resources (called before run loop starts).
        Must be implemented by derived agents.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def execute(self):
        """
        The core logic of the agent.
        This will be executed in a loop while the agent is running.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def teardown(self):
        """
        Cleanup phase for releasing resources (called after stop).
        Must be implemented by derived agents.
        """
        raise NotImplementedError

    def start(self):
        """
        Start the agent in a dedicated thread.
        """
        if self.running:
            self.logger.warning(f"Agent {self.agent_id} is already running.")
            return

        self.logger.info(f"Starting agent {self.agent_id}...")
        self.running = True

        try:
            self.setup()
        except Exception as e:
            self.logger.error(f"Setup failed for {self.agent_id}: {e}", exc_info=True)
            self.running = False
            return

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        """
        Internal run loop for the agent.
        """
        while self.running:
            try:
                self.execute()
            except Exception as e:
                self.logger.error(f"Error in agent {self.agent_id}: {e}", exc_info=True)

            time.sleep(self.config.get("interval", 1))  # Default interval 1s

        try:
            self.teardown()
        except Exception as e:
            self.logger.error(f"Teardown failed for {self.agent_id}: {e}", exc_info=True)

        self.logger.info(f"Agent {self.agent_id} stopped.")

    def stop(self):
        """
        Stop the agent gracefully.
        """
        if not self.running:
            self.logger.warning(f"Agent {self.agent_id} is not running.")
            return

        self.logger.info(f"Stopping agent {self.agent_id}...")
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def get_status(self) -> Dict[str, Any]:
        """
        Return the current status of the agent.
        """
        return {
            "agent_id": self.agent_id,
            "class": self.__class__.__name__,
            "running": self.running,
            "config": self.config,
        }
import torch
import random
import numpy as np
from cyberbattle.agents.baseline.agent_wrapper import AgentWrapper


class SentenialAgent:
    """Wrapper for baseline RL agent in CyberBattleSim."""

    def __init__(self, environment, episodes=1000):
        self.environment = environment
        self.episodes = episodes
        self.agent = AgentWrapper(environment)

    def train(self):
        print(f"[+] Training Sentenial-X agent for {self.episodes} episodes...")
        self.agent.train(episodes=self.episodes)

    def act(self, state):
        """Custom decision logic can be added here later."""
        return self.agent.exploit(state)
