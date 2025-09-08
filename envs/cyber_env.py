# envs/cyber_env.py
import os
import logging

logger = logging.getLogger("SentenialX.Env")

# Attempt to import CyberBattleSim; if not present, fall back to a minimal dummy env for testing
try:
    from cyberbattle.simulation import env as cyber_env_module
    HAS_CYBER = True
except Exception:
    HAS_CYBER = False

# Gym fallback for test
import gym
import numpy as np


def create_environment():
    """
    Returns a gym-compatible environment appropriate for RL training.
    If CyberBattleSim is available it will instantiate a default scenario.
    Otherwise, a simple dummy env is returned (for CI / test).
    """
    if HAS_CYBER:
        logger.info("Creating CyberBattleSim environment")
        # CyberBattleEnv is the common entry — adapt if your version differs
        environment = cyber_env_module.CyberBattleEnv()
        return environment
    else:
        logger.warning("CyberBattleSim not found — returning dummy Gym environment for testing")

        class DummyEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.observation_space = gym.spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
                self.action_space = gym.spaces.Discrete(4)
                self._step = 0

            def reset(self):
                self._step = 0
                return np.zeros(self.observation_space.shape, dtype=np.float32)

            def step(self, action):
                self._step += 1
                obs = np.random.rand(*self.observation_space.shape).astype(np.float32)
                reward = float(np.random.randn()) * 0.1
                done = self._step >= 50
                info = {}
                return obs, reward, done, info

            def render(self, mode="human"):
                pass

        return DummyEnv()
