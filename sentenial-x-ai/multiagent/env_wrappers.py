"""
Environment Wrappers for Multi-Agent Training
=============================================

Provides wrappers around single-agent cyber defense environments
(e.g., CyberBattleSim, CybORG) to enable multi-agent reinforcement
learning (MARL).

Key Features:
- Multiple agents interacting in the same environment
- Shared or independent observations
- Configurable reward sharing
- Gym-compatible API
"""

import gym
import numpy as np
from typing import Dict, List, Any


class MultiAgentEnv(gym.Env):
    """
    A generic multi-agent wrapper for OpenAI Gym-style environments.
    Turns a single-agent environment into N agents that take turns or
    act simultaneously depending on the mode.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, base_env: gym.Env, agent_ids: List[str], shared_rewards: bool = False):
        super().__init__()
        self.base_env = base_env
        self.agent_ids = agent_ids
        self.shared_rewards = shared_rewards

        # Each agent sees the same observation/space for simplicity
        self.observation_space = base_env.observation_space
        self.action_space = base_env.action_space

        # Track per-agent state
        self._agent_states: Dict[str, Any] = {}

    def reset(self) -> Dict[str, np.ndarray]:
        obs = self.base_env.reset()
        self._agent_states = {agent_id: obs for agent_id in self.agent_ids}
        return self._agent_states

    def step(self, actions: Dict[str, Any]) -> (Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, Any]):
        """
        Perform a multi-agent step.

        :param actions: Dict of {agent_id: action}
        :return: obs, rewards, dones, infos (all dicts keyed by agent_id)
        """
        obs, reward, done, info = self.base_env.step(actions[self.agent_ids[0]])

        # Share or split rewards
        rewards = {aid: (reward if self.shared_rewards else float(np.random.choice([0, reward]))) for aid in self.agent_ids}
        observations = {aid: obs for aid in self.agent_ids}
        dones = {aid: done for aid in self.agent_ids}
        infos = {aid: info for aid in self.agent_ids}

        return observations, rewards, dones, infos

    def render(self, mode="human"):
        return self.base_env.render(mode=mode)

    def close(self):
        self.base_env.close()


class CooperativeDefenseWrapper(MultiAgentEnv):
    """
    Wrapper for cooperative multi-agent training:
    - All defenders share the same reward signal
    - Designed for MARL defense scenarios
    """

    def __init__(self, base_env: gym.Env, num_defenders: int = 2):
        agent_ids = [f"defender_{i}" for i in range(num_defenders)]
        super().__init__(base_env, agent_ids, shared_rewards=True)


class CompetitiveDefenseWrapper(MultiAgentEnv):
    """
    Wrapper for competitive training:
    - Defender vs. Attacker agent(s)
    - Each agent gets its own reward signal
    """

    def __init__(self, base_env: gym.Env, num_attackers: int = 1, num_defenders: int = 1):
        agent_ids = [f"attacker_{i}" for i in range(num_attackers)] + [f"defender_{j}" for j in range(num_defenders)]
        super().__init__(base_env, agent_ids, shared_rewards=False)
