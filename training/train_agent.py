import logging
from agents.baseline_agent import SentenialAgent
from envs.cyber_env import create_environment
from utils.logger import log_metrics

def train_agent(episodes: int = 100) -> None:
    """
    Train the SentenialAgent on the provided environment.

    Args:
        episodes (int): Number of training episodes.
    """
    # Initialize environment and agent
    env = create_environment()
    agent = SentenialAgent(env, episodes=episodes)

    logging.info(f"Starting training for {episodes} episodes.")
    agent.train()
    logging.info("Training completed.")

    # Save metrics
    log_metrics({"episodes": episodes, "status": "completed"})
    logging.info("Metrics logged successfully.")

def main():
    train_agent(episodes=100)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    main()
