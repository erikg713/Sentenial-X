# agents/dqn_agent.py
import os
import time
import math
import random
import logging
from typing import Deque, Tuple, List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque, namedtuple

from agents.base_agent import BaseAgent

# Simple transition tuple
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buffer: Deque[Transition] = deque(maxlen=self.capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DQNAgent(BaseAgent):
    """
    DQNAgent implements a Deep Q-Network that fits into the BaseAgent lifecycle.
    It expects the environment to follow OpenAI Gym API:
      - env.reset() -> observation (numpy)
      - env.step(action) -> (next_obs, reward, done, info)
      - env.action_space.n or env.action_space.sample()
      - env.observation_space.shape (or you can pass obs_dim explicitly)
    """

    def __init__(self, agent_id: str, env, config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, config)
        self.env = env

        # Default hyperparameters; override via config dict
        cfg = self.config
        self.device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.obs_dim = cfg.get("obs_dim", None)  # can be None -> infer in setup
        self.action_dim = cfg.get("action_dim", None)  # can be None -> infer in setup

        self.hidden_dims = cfg.get("hidden_dims", [256, 256])
        self.lr = float(cfg.get("lr", 1e-4))
        self.gamma = float(cfg.get("gamma", 0.99))
        self.batch_size = int(cfg.get("batch_size", 64))
        self.replay_capacity = int(cfg.get("replay_capacity", 100000))
        self.min_replay_size = int(cfg.get("min_replay_size", 1000))
        self.target_update_tau = float(cfg.get("target_update_tau", 0.005))  # soft update
        self.target_update_every = int(cfg.get("target_update_every", 1))  # also can use hard update
        self.epsilon_start = float(cfg.get("epsilon_start", 1.0))
        self.epsilon_final = float(cfg.get("epsilon_final", 0.02))
        self.epsilon_decay = float(cfg.get("epsilon_decay", 100000))  # steps for linear decay
        self.max_training_steps = int(cfg.get("max_training_steps", 200000))
        self.train_freq = int(cfg.get("train_freq", 1))  # how often to call train_step per execute
        self.save_path = cfg.get("save_path", f"checkpoints/{self.agent_id}.pth")
        self.seed = int(cfg.get("seed", 0))

        # internal state
        self.replay = ReplayBuffer(self.replay_capacity)
        self.policy_net: Optional[QNetwork] = None
        self.target_net: Optional[QNetwork] = None
        self.optimizer: Optional[optim.Optimizer] = None

        self.total_steps = 0
        self.episode_rewards: List[float] = []
        self.current_episode_reward = 0.0
        self.episode_length = 0
        self.episodes = int(cfg.get("episodes", 0))  # used in training scenarios
        self.max_episode_steps = int(cfg.get("max_episode_steps", 1000))
        self.seed_everything(self.seed)

    # -----------------------
    # Helpers
    # -----------------------
    def seed_everything(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def infer_dims_from_env(self):
        # try to infer obs dim and action dim from env
        if self.obs_dim is None:
            try:
                obs_space = getattr(self.env, "observation_space", None)
                if obs_space is not None:
                    self.obs_dim = int(np.prod(obs_space.shape))
                else:
                    # fallback: run a reset and inspect
                    obs = self.env.reset()
                    self.obs_dim = int(np.prod(np.array(obs).shape))
            except Exception:
                raise ValueError("Could not infer observation dimension. Please set obs_dim in config.")

        if self.action_dim is None:
            try:
                action_space = getattr(self.env, "action_space", None)
                if action_space is not None and hasattr(action_space, "n"):
                    self.action_dim = int(action_space.n)
                else:
                    # try sample to determine discrete count (not guaranteed)
                    sample = self.env.action_space.sample()
                    if isinstance(sample, (int, np.integer)):
                        # unknown n, user must provide
                        raise ValueError("action_space has no 'n'. Please set action_dim in config.")
                    else:
                        raise ValueError("Unsupported action_space type. Provide action_dim in config.")
            except Exception:
                raise ValueError("Could not infer action dimension. Please set action_dim in config.")

    def build_networks(self):
        assert self.obs_dim is not None and self.action_dim is not None
        self.policy_net = QNetwork(self.obs_dim, self.action_dim, self.hidden_dims).to(self.device)
        self.target_net = QNetwork(self.obs_dim, self.action_dim, self.hidden_dims).to(self.device)
        # copy params
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def epsilon_by_frame(self, frame_idx: int) -> float:
        # linear decay
        eps = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * max(
            0.0, (1.0 - frame_idx / max(1.0, self.epsilon_decay))
        )
        return float(eps)

    # -----------------------
    # DQN Core Methods
    # -----------------------
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if eval_mode:
            with torch.no_grad():
                qvals = self.policy_net(state_tensor)
                action = int(qvals.argmax(dim=1).item())
            return action

        eps_threshold = self.epsilon_by_frame(self.total_steps)
        if random.random() < eps_threshold:
            # random action
            try:
                a = self.env.action_space.sample()
                if isinstance(a, (int, np.integer)):
                    return int(a)
                # if action_space.sample returns vector, fallback to random integer
            except Exception:
                pass
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                qvals = self.policy_net(state_tensor)
                action = int(qvals.argmax(dim=1).item())
            return action

    def compute_td_loss(self, batch: List[Transition]) -> torch.Tensor:
        states = torch.FloatTensor(np.stack([np.array(t.state) for t in batch])).to(self.device)
        actions = torch.LongTensor([int(t.action) for t in batch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([float(t.reward) for t in batch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.stack([np.array(t.next_state) for t in batch])).to(self.device)
        dones = torch.FloatTensor([float(t.done) for t in batch]).unsqueeze(1).to(self.device)

        # current Q
        q_values = self.policy_net(states).gather(1, actions)  # (B,1)

        # target Q
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)  # (B,1)
            target_q = rewards + (1.0 - dones) * (self.gamma * next_q_values)

        loss = nn.functional.mse_loss(q_values, target_q)
        return loss

    def soft_update(self, tau: float):
        # θ_target = τ*θ_local + (1-τ)*θ_target
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, path: Optional[str] = None):
        path = path or self.save_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "policy_state": self.policy_net.state_dict(),
                "target_state": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
            },
            path,
        )
        self.logger.info(f"Saved DQN checkpoint to {path}")

    def load(self, path: Optional[str] = None):
        path = path or self.save_path
        if not os.path.exists(path):
            self.logger.warning(f"No checkpoint found at {path}")
            return
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_state"])
        self.target_net.load_state_dict(ckpt["target_state"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = int(ckpt.get("total_steps", 0))
        self.logger.info(f"Loaded DQN checkpoint from {path}")

    # -----------------------
    # BaseAgent lifecycle impl
    # -----------------------
    def setup(self):
        """
        Called once before the run loop. Builds networks and populates a small initial replay buffer
        by taking random actions if configured to do so.
        """
        self.logger.info("DQN setup starting...")
        self.infer_dims_from_env()
        self.build_networks()
        self.logger.info(f"Device: {self.device}, obs_dim: {self.obs_dim}, action_dim: {self.action_dim}")

        # Optional: prefill replay buffer with random trajectories
        prefill = int(self.config.get("prefill_replay", 1000))
        if prefill > 0:
            self.logger.info(f"Prefilling replay buffer with {prefill} transitions...")
            obs = self.env.reset()
            for _ in range(prefill):
                a = self.env.action_space.sample() if hasattr(self.env, "action_space") else random.randrange(self.action_dim)
                next_obs, reward, done, _ = self.env.step(a)
                self.replay.push(obs, a, reward, next_obs, done)
                obs = next_obs if not done else self.env.reset()
            self.logger.info(f"Prefill complete. Replay size: {len(self.replay)}")

    def execute(self):
        """
        Core run loop executed repeatedly by BaseAgent._run.
        This function performs environment interaction and training. It respects
        train_freq to avoid training on every tick if not wanted.
        """
        # If episodes configured, run a single episode per execute invocation (or chunked)
        self.logger.debug("DQN execute step")
        # Play one episode (or partial, controlled by max_episode_steps)
        state = self.env.reset()
        episode_reward = 0.0
        for step in range(self.max_episode_steps):
            action = self.select_action(state, eval_mode=False)
            next_state, reward, done, info = self.env.step(action)

            self.replay.push(state, action, reward, next_state, done)
            self.total_steps += 1
            episode_reward += float(reward)

            # training step
            if len(self.replay) >= self.min_replay_size and (self.total_steps % self.train_freq == 0):
                batch = self.replay.sample(self.batch_size)
                loss = self.compute_td_loss(batch)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
                self.optimizer.step()

                # soft update target network
                if self.target_update_tau is not None:
                    self.soft_update(self.target_update_tau)
                # or use hard update on interval
                if self.target_update_every and (self.total_steps % (self.target_update_every * 1000) == 0):
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            state = next_state
            if done:
                break

        self.current_episode_reward = episode_reward
        self.episode_rewards.append(episode_reward)
        self.logger.info(f"Episode finished. Reward: {episode_reward:.3f}, total_steps: {self.total_steps}, replay_size: {len(self.replay)}")

        # checkpointing
        if self.total_steps % max(1, int(self.config.get("checkpoint_every_steps", 10000))) == 0:
            self.save()

    def teardown(self):
        self.logger.info("DQN teardown: saving model and cleaning up...")
        try:
            self.save()
        except Exception as e:
            self.logger.error(f"Error saving model during teardown: {e}", exc_info=True)

    # -----------------------
    # Utility methods
    # -----------------------
    def evaluate(self, episodes: int = 5, render: bool = False) -> float:
        """
        Run evaluation episodes using greedy policy (no exploration).
        Returns average reward.
        """
        total = 0.0
        for ep in range(episodes):
            obs = self.env.reset()
            ep_reward = 0.0
            for _ in range(self.max_episode_steps):
                action = self.select_action(obs, eval_mode=True)
                obs, reward, done, _ = self.env.step(action)
                ep_reward += float(reward)
                if render:
                    try:
                        self.env.render()
                    except Exception:
                        pass
                if done:
                    break
            total += ep_reward
        avg = total / float(episodes)
        self.logger.info(f"Evaluation avg reward over {episodes} episodes: {avg:.3f}")
        return avg


# Example usage snippet (run outside of class, e.g. in training script)
if __name__ == "__main__":
    # Minimal example showing how to start the agent in standalone mode
    from envs.cyber_env import create_environment

    env = create_environment()
    config = {
        "obs_dim": None,  # infer from env
        "action_dim": None,  # infer from env
        "hidden_dims": [256, 256],
        "lr": 1e-4,
        "batch_size": 64,
        "replay_capacity": 50000,
        "min_replay_size": 1000,
        "prefill_replay": 2000,
        "epsilon_start": 1.0,
        "epsilon_final": 0.02,
        "epsilon_decay": 100000,
        "train_freq": 1,
        "max_episode_steps": 500,
        "checkpoint_every_steps": 20000,
        "save_path": "checkpoints/dqn_agent.pth",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 0,
    }

    agent = DQNAgent("dqn-01", env, config)
    agent.start()

    # Let the agent run for some seconds (or manage via external lifecycle)
    try:
        time.sleep(60 * 10)  # run for 10 minutes (example)
    except KeyboardInterrupt:
        pass
    finally:
        agent.stop()
        agent.evaluate(episodes=3, render=False)
