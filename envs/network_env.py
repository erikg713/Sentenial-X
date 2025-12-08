# envs/network_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
import random
import time

logger = logging.getLogger("SentenialX.Env")

# Define network states for observation encoding
STATE_NORMAL = 0
STATE_PROBING = 1
STATE_LATERAL = 2
STATE_EXFIL = 3

class NetworkTrafficEnv(gym.Env):
    """
    A custom Gymnasium environment for training an RL countermeasure agent 
    to detect adversarial network activity.
    """
    metadata = {'render_modes': ['human', 'state'], 'render_fps': 5}

    def __init__(self, attack_difficulty=0.2, max_steps=100):
        super().__init__()
        
        # Observation Space: (6 features)
        # [0: Pkt_Rate, 1: Flow_Ratio_In_Out, 2: Host_CPU, 3: Suspicious_Conn_Count, 4: Current_Attack_State, 5: Time_Since_Alert]
        self.observation_space = spaces.Box(low=0, high=100, shape=(6,), dtype=np.float32)
        
        # Action Space: 4 discrete actions the agent can take
        # 0: NO_ACTION (Monitor), 1: ISOLATE_HOST, 2: BLOCK_FLOW, 3: RAISE_ALERT
        self.action_space = spaces.Discrete(4)
        
        self.attack_difficulty = attack_difficulty
        self.max_steps = max_steps
        self.current_step = 0
        self.current_state = STATE_NORMAL
        self.is_attack_active = False
        logger.info("NetworkTrafficEnv initialized.")

    def _get_obs(self):
        """Generates the observation array based on the current state."""
        obs = np.array([
            random.uniform(10, 50) + self._get_noise(0),  # Pkt Rate
            random.uniform(0.5, 1.5) + self._get_noise(1), # Flow Ratio
            random.uniform(5, 40) + self._get_noise(2),   # Host CPU
            random.uniform(0, 5) + self._get_noise(3),    # Suspicious Connections
            self.current_state,
            (time.time() - self.start_time) # Time elapsed (normalized later)
        ], dtype=np.float32)
        
        # Normalize continuous features to [0, 100] scale for simplicity
        obs[0:4] = np.clip(obs[0:4], 0, 100) 
        
        return obs

    def _get_noise(self, feature_index):
        """Adds signal (noise) to features based on the current attack state."""
        # When in an attack state, specific features get boosted
        if self.current_state == STATE_PROBING:
            # Probing increases pkt rate (index 0) and suspicious connection count (index 3)
            return random.uniform(20, 50) if feature_index in [0, 3] else 0
        elif self.current_state == STATE_EXFIL:
            # Exfil increases flow ratio (index 1) and host CPU (index 2)
            return random.uniform(30, 70) if feature_index in [1, 2] else 0
        return 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_state = STATE_NORMAL
        self.is_attack_active = False
        self.attack_start_step = random.randint(10, 30)
        self.start_time = time.time()
        
        observation = self._get_obs()
        info = {}
        logger.info("Network Environment reset.")
        return observation, info

    def step(self, action):
        self.current_step += 1
        reward = -0.01  # Small penalty for every step (encourages speed)
        terminated = False
        truncated = self.current_step >= self.max_steps
        info = {}

        # 1. Update Attack State
        if self.current_step == self.attack_start_step and not self.is_attack_active:
            self.is_attack_active = True
            self.current_state = STATE_PROBING
            logger.info(f"Attack initiated at step {self.current_step}")
        
        if self.is_attack_active and self.current_state == STATE_PROBING and self.current_step > self.attack_start_step + 15:
            # Attack moves from probing to lateral/exfil stage
            self.current_state = STATE_EXFIL
            logger.info("Attack escalated to Exfiltration.")

        # 2. Process Agent Action
        if self.current_state != STATE_NORMAL and self.is_attack_active:
            # High reward for correct blocking action (ISOLATE_HOST or BLOCK_FLOW)
            if action in [1, 2]:
                reward += 10.0
                terminated = True
                info['outcome'] = 'Threat Mitigated'
                logger.warning(f"Threat mitigated by Action {action} at step {self.current_step}!")
            # Penalty for alerting without high confidence
            elif action == 3 and self.current_state == STATE_PROBING:
                reward -= 1.0 # False alarm cost
        
        # 3. Process Failed Defense (Attack success)
        if self.current_state == STATE_EXFIL and self.current_step > self.attack_start_step + 40 and not terminated:
            reward -= 20.0 # Large penalty for successful exfiltration
            terminated = True
            info['outcome'] = 'Threat Failed to Mitigate'
            logger.error("Exfiltration complete. Threat mitigated failed.")

        observation = self._get_obs()
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.current_step % 10 == 0:
            print(f"[{self.current_step}/{self.max_steps}] State: {self.current_state} | Reward: {reward:.2f}")

    def close(self):
        pass
