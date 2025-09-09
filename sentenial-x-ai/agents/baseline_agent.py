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
      
