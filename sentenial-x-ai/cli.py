#!/usr/bin/env python3
"""
Sentenial-X AI CLI
==================
Command-line interface to manage and run Sentenial-X autonomous agents.

Usage Examples:
---------------
# Train a baseline agent
python cli.py train --episodes 1000

# Run agent in simulation
python cli.py run --agent-id agent001

# Show agent status
python cli.py status --agent-id agent001
"""

import argparse
import logging
from sentenial_x.agents.base_agent import BaseAgent
from sentenial_x.agents.sentenial_agent import SentenialAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("SentenialX.CLI")


def main():
    parser = argparse.ArgumentParser(description="Sentenial-X AI CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    parser_train = subparsers.add_parser("train", help="Train an RL agent")
    parser_train.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")

    # Run command
    parser_run = subparsers.add_parser("run", help="Run an agent in simulation")
    parser_run.add_argument("--agent-id", type=str, required=True, help="Agent ID to run")
    parser_run.add_argument("--interval", type=float, default=1.0, help="Execution interval in seconds")

    # Status command
    parser_status = subparsers.add_parser("status", help="Get agent status")
    parser_status.add_argument("--agent-id", type=str, required=True, help="Agent ID to query")

    args = parser.parse_args()

    if args.command == "train":
        # Initialize default CyberBattleSim environment
        try:
            from cyberbattle.simulation import env
        except ImportError:
            logger.error("CyberBattleSim not installed. Install with `pip install CyberBattleSim`.")
            return

        environment = env.CyberBattleEnv()
        agent = SentenialAgent(environment, episodes=args.episodes)
        agent.train()

    elif args.command == "run":
        # For simplicity, using a SentenialAgent wrapper
        from cyberbattle.simulation import env
        environment = env.CyberBattleEnv()
        agent = SentenialAgent(environment)
        agent_wrapper = agent.agent  # Access internal AgentWrapper
        logger.info(f"Running agent {args.agent_id} with interval {args.interval}s...")

        try:
            while True:
                state = environment.get_state()
                action = agent.act(state)
                environment.step(action)
        except KeyboardInterrupt:
            logger.info("Execution interrupted by user.")

    elif args.command == "status":
        # Placeholder for real agent status retrieval
        logger.info(f"Querying status for agent {args.agent_id}...")
        # In full implementation, retrieve status from orchestrator or registry
        print({"agent_id": args.agent_id, "status": "unknown", "running": False})

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
