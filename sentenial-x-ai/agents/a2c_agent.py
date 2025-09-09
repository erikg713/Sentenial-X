from stable_baselines3 import A2C
from agents.base_agent import BaseAgent
from stable_baselines3.common.vec_env import DummyVecEnv

class A2CAgent(BaseAgent):
    def setup(self):
        n_envs = int(self.config.get("n_envs", 2))
        self.vec_env = DummyVecEnv([self.env_fn for _ in range(n_envs)])
        self.model = A2C("MlpPolicy", self.vec_env, verbose=1)
    def execute(self):
        self.model.learn(total_timesteps=self.config.get("timesteps_per_execute", 10000))
    def teardown(self):
        self.vec_env.close()
