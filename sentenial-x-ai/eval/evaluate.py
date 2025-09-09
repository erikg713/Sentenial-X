"""
Sentenial-X AI Evaluation Module
================================

Provides evaluation routines for RL agents, including:
- Simulation rollouts
- Reward aggregation
- Performance metrics
- Logging and reporting
"""

import logging
from typing import Dict, Any
from sentenial_x.config import Config
from sentenial_x.agents.sentenial_agent import SentenialAgent

logger = logging.getLogger("SentenialX.Eval")
logger.setLevel(Config.LOG_LEVEL)


class Evaluator:
    """
    Evaluator class for testing RL agents in CyberBattleSim environments.
    """

    def __init__(self, agent: SentenialAgent, environment, episodes: int = None):
        """
        Initialize evaluator with agent and environment.

        :param agent: SentenialAgent instance
        :param environment: Gym-like environment
        :param episodes: Number of evaluation episodes
        """
        self.agent = agent
        self.env = environment
        self.episodes = episodes or Config.CYBERBATTLE_MAX_EPISODES

    def evaluate(self) -> Dict[str, Any]:
        """
        Run evaluation loop over defined number of episodes.
        Returns performance metrics.
        """
        logger.info(f"Starting evaluation for {self.episodes} episodes...")
        total_rewards = []
        success_count = 0

        for episode in range(1, self.episodes + 1):
            state = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.agent.act(state)
                state, reward, done, info = self.env.step(action)
                episode_reward += reward

            total_rewards.append(episode_reward)
            if self._episode_success(info):
                success_count += 1

            logger.info(f"Episode {episode} completed: Reward={episode_reward}")

        avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0
        success_rate = success_count / self.episodes

        metrics = {
            "total_episodes": self.episodes,
            "average_reward": avg_reward,
            "success_rate": success_rate,
            "total_successes": success_count,
        }

        logger.info(f"Evaluation complete: {metrics}")
        return metrics

    @staticmethod
    def _episode_success(info: dict) -> bool:
        """
        Determine if episode was successful based on environment info.
        Customize based on CyberBattleSim success criteria.
        """
        return info.get("success", False)


# Example usage
if __name__ == "__main__":
    import gym
    try:
        from cyberbattle.simulation import env
    except ImportError:
        logger.error("CyberBattleSim not installed. Install with `pip install CyberBattleSim`.")
        exit(1)

    environment = env.CyberBattleEnv()
    agent_wrapper = SentenialAgent(environment)
    evaluator = Evaluator(agent_wrapper, environment, episodes=10)
    results = evaluator.evaluate()
    print(results)
