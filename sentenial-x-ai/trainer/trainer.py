import os
import time
import logging
import torch
from typing import Dict, Any, Optional

logger = logging.getLogger("sentenial.trainer")


class Trainer:
    """
    Core training loop for Sentenial-X AI agents.
    Supports checkpointing, telemetry integration, and modular RL backends.
    """

    def __init__(
        self,
        agent,
        environment,
        output_dir: str = "checkpoints",
        episodes: int = 1000,
        max_steps: int = 500,
        device: Optional[str] = None,
        telemetry_client=None,
    ):
        self.agent = agent
        self.environment = environment
        self.episodes = episodes
        self.max_steps = max_steps
        self.output_dir = output_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.telemetry_client = telemetry_client

        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"[Trainer] Initialized on device={self.device}")

    def train(self):
        """
        Main training loop: episodes x steps with logging + optional telemetry.
        """
        rewards_history = []

        for episode in range(1, self.episodes + 1):
            state = self.environment.reset()
            total_reward = 0

            for step in range(self.max_steps):
                action = self.agent.act(state)
                next_state, reward, done, _ = self.environment.step(action)

                self.agent.learn(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if done:
                    break

            rewards_history.append(total_reward)
            avg_reward = sum(rewards_history[-50:]) / min(len(rewards_history), 50)

            logger.info(
                f"[Trainer] Episode {episode}/{self.episodes} "
                f"Reward={total_reward:.2f} AvgReward={avg_reward:.2f}"
            )

            # Send telemetry update if enabled
            if self.telemetry_client:
                import asyncio
                asyncio.run(
                    self.telemetry_client.send_event(
                        "training_progress",
                        {
                            "episode": episode,
                            "reward": total_reward,
                            "avg_reward": avg_reward,
                        },
                    )
                )

            # Save checkpoint every 50 episodes
            if episode % 50 == 0:
                self.save_checkpoint(episode)

        logger.info("[Trainer] Training complete")

    def save_checkpoint(self, episode: int):
        """
        Save agent model checkpoint.
        """
        path = os.path.join(self.output_dir, f"agent_ep{episode}.pt")
        try:
            self.agent.save(path)
            logger.info(f"[Trainer] Saved checkpoint: {path}")
        except Exception as e:
            logger.error(f"[Trainer] Failed to save checkpoint: {e}")

    def evaluate(self, episodes: int = 10) -> Dict[str, Any]:
        """
        Run evaluation episodes and return stats.
        """
        rewards = []
        for _ in range(episodes):
            state = self.environment.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.agent.act(state, exploit=True)
                state, reward, done, _ = self.environment.step(action)
                total_reward += reward
            rewards.append(total_reward)

        avg_reward = sum(rewards) / episodes
        logger.info(f"[Trainer] Evaluation: AvgReward={avg_reward:.2f}")
        return {"avg_reward": avg_reward, "episodes": episodes}
