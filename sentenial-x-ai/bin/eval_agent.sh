#!/usr/bin/env bash
# Evaluate a trained agent using evaluator module
set -euo pipefail

EPISODES="${EPISODES:-10}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/ppo_main.zip}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

export PYTHONUNBUFFERED=1
echo "[+] Activating venv if present..."
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python - <<PY
import logging
from sentenial_x.eval.evaluate import Evaluator
from sentenial_x.agents.sentenial_agent import SentenialAgent
from envs.cyber_env import create_environment
logging.basicConfig(level=logging.${LOG_LEVEL})
env = create_environment()
# instantiate agent wrapper - if you have a PPO wrapper with load(), use that instead
from agents.ppo_agent import PPOAgent
agent = PPOAgent("eval-ppo", lambda: create_environment(), config={"n_envs":1})
agent.setup()
try:
    agent.load("${CHECKPOINT_PATH}")
except Exception as e:
    print("Warning: failed to load checkpoint:", e)
evaluator = Evaluator(agent, env, episodes=${EPISODES})
metrics = evaluator.evaluate()
print("Evaluation metrics:", metrics)
PY
