# training/train_ppo.py
import os
import argparse
import logging
from typing import Callable
from agents.ppo_agent import PPOAgent
from envs.cyber_env import create_environment
from utils.logger import setup_logging

logger = logging.getLogger("training.train_ppo")


def env_fn_factory():
    # Return a callable for SB3 vec env creation
    def _fn():
        return create_environment()
    return _fn


def main(args):
    setup_logging(level=args.log_level)

    env_fn = env_fn_factory()
    config = {
        "n_envs": args.n_envs,
        "policy": args.policy,
        "total_timesteps": args.total_timesteps,
        "tensorboard_log": args.tb_log,
        "timesteps_per_execute": args.chunk,
        "save_path": args.save_path,
        "checkpoint_interval_steps": args.checkpoint_interval,
    }

    agent = PPOAgent("ppo-main", env_fn, config=config)
    agent.start()
    try:
        # Wait until agent stops itself (when timesteps exhausted) or external stop
        while agent.running:
            agent._thread.join(timeout=5)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — stopping agent...")
    finally:
        agent.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--policy", type=str, default="MlpPolicy")
    p.add_argument("--total-timesteps", type=int, default=250_000)
    p.add_argument("--chunk", type=int, default=50_000)
    p.add_argument("--tb-log", type=str, default="runs/ppo_main")
    p.add_argument("--save-path", type=str, default="checkpoints/ppo_main.zip")
    p.add_argument("--checkpoint-interval", type=int, default=50_000)
    p.add_argument("--log-level", type=str, default="INFO")
    args = p.parse_args()
    main(args)
