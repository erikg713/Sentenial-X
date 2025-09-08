from agents.baseline_agent import SentenialAgent
from envs.cyber_env import create_environment
from utils.logger import log_metrics


def main():
    # Load environment
    environment = create_environment()

    # Initialize agent
    agent = SentenialAgent(environment, episodes=100)

    # Train agent
    agent.train()

    # Save logs
    log_metrics({"episodes": 100, "status": "completed"})


if __name__ == "__main__":
    main()
