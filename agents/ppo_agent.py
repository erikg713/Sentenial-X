# agents/ppo_agent.py
from typing import Optional, Callable, Dict, Any
import os
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from agents.base_agent import BaseAgent

logger = logging.getLogger("SentenialX.PPOAgent")


class PPOAgent(BaseAgent):
    """
    PPO Agent wrapper that conforms to BaseAgent lifecycle.
    Expects env_fn: Callable[[], gym.Env] that returns a fresh env instance.
    """

    def __init__(self, agent_id: str, env_fn: Callable[[], object], config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, config)
        self.env_fn = env_fn
        self.model: Optional[PPO] = None
        self.vec_env = None

        # load config defaults
        self.n_envs = int(self.config.get("n_envs", 4))
        self.policy = self.config.get("policy", "MlpPolicy")
        self.total_timesteps = int(self.config.get("total_timesteps", 250_000))
        self.tb_log = self.config.get("tensorboard_log", None)
        self.save_path = self.config.get("save_path", f"checkpoints/ppo_{self.agent_id}.zip")
        self.checkpoint_interval = int(self.config.get("checkpoint_interval_steps", 50_000))
        self.learn_kwargs = self.config.get("learn_kwargs", {})

    def setup(self):
        self.logger.info(f"PPOAgent.setup: creating {self.n_envs} vectorized env(s).")
        if self.n_envs <= 1:
            self.vec_env = DummyVecEnv([self.env_fn])
        else:
            # SubprocVecEnv for parallel envs (safer for CPU-bound envs)
            self.vec_env = SubprocVecEnv([self.env_fn for _ in range(self.n_envs)])

        os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
        self.model = PPO(self.policy, self.vec_env, verbose=1, tensorboard_log=self.tb_log, **self.learn_kwargs)
        self.logger.info("PPO model initialized")

    def execute(self):
        """
        Called repeatedly by BaseAgent._run. For training we run a chunk of timesteps per execute.
        """
        chunk = int(self.config.get("timesteps_per_execute", min(50_000, self.total_timesteps)))
        remaining = max(0, self.total_timesteps - getattr(self, "_trained_steps", 0))
        run_for = min(chunk, remaining)
        if run_for <= 0:
            self.logger.info("PPOAgent: training complete (no remaining timesteps).")
            self.stop()
            return

        self.logger.info(f"PPOAgent training chunk: {run_for} timesteps (remaining {remaining})")
        # `learn` will block until run_for steps are collected
        self.model.learn(total_timesteps=run_for, reset_num_timesteps=False)
        self._trained_steps = getattr(self, "_trained_steps", 0) + run_for

        # checkpointing
        if self._trained_steps % self.checkpoint_interval == 0 or self._trained_steps >= self.total_timesteps:
            self.logger.info(f"PPOAgent saving checkpoint at {self._trained_steps} steps -> {self.save_path}")
            self.model.save(self.save_path)

    def teardown(self):
        try:
            if self.vec_env is not None:
                self.vec_env.close()
            if self.model is not None:
                self.model.save(self.save_path)
                self.logger.info(f"PPOAgent teardown: saved model to {self.save_path}")
        except Exception as e:
            self.logger.exception("Error during PPOAgent.teardown: %s", e)

    def act(self, observation, deterministic=True):
        if self.model is None:
            raise RuntimeError("PPO model not initialized")
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def load(self, path: str):
        self.logger.info(f"PPOAgent loading model from {path}")
        self.model = PPO.load(path, env=self.vec_env)
