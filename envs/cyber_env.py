# envs/cyber_env.py

import os
import logging
import gymnasium as gym 
from gymnasium import spaces
import numpy as np

# Logger name corrected to SentenialX.Env
logger = logging.getLogger("SentenialX.Env") 

# Attempt to import CyberBattleSim
try:
    from cyberbattle.simulation import env as cyber_env_module
    HAS_CYBER = True
except ImportError:
    HAS_CYBER = False

def create_environment():
    """
    Returns a gym-compatible environment appropriate for RL training.
    """
    # This block requires CyberBattleSim installation
    if HAS_CYBER:
        logger.info("Creating CyberBattleSim environment.")
        # CyberBattleEnv is the common entry
        environment = cyber_env_module.CyberBattleEnv()
        return environment
    else:
        logger.warning("CyberBattleSim not found — returning dummy Gymnasium environment for testing.")

        # Define a Gymnasium-compatible dummy environment
        class DummyEnv(gym.Env):
            metadata = {'render_modes': ['human'], 'render_fps': 30}
            
            def __init__(self):
                super().__init__()
                self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
                self.action_space = spaces.Discrete(4)
                self._step = 0

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                self._step = 0
                observation = np.zeros(self.observation_space.shape, dtype=np.float32)
                info = {}
                return observation, info

            def step(self, action):
                self._step += 1
                obs = self.np_random.random(self.observation_space.shape).astype(np.float32)
                reward = float(self.np_random.standard_normal()) * 0.1
                terminated = self._step >= 50
                truncated = False 
                info = {}
                return obs, reward, terminated, truncated, info

            def render(self):
                pass

            def close(self):
                pass

        return DummyEnv()
