#!/usr/bin/env bash
# Run a simple orchestrator demonstration (blocks)
set -euo pipefail

LOG_LEVEL="${LOG_LEVEL:-INFO}"
export SENTENIAL_LOG_LEVEL="${LOG_LEVEL}"

echo "[+] Activating venv if present..."
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python - <<'PY'
from sentenial_x.orchestrator import Orchestrator
from sentenial_x.agents.sentenial_agent import SentenialAgent
from envs.cyber_env import create_environment
import time, logging
logging.basicConfig(level=logging.INFO)
o = Orchestrator()
env = create_environment()
agent_wrapper = SentenialAgent(env, episodes=10)
# register the internal AgentWrapper (compat with earlier structure)
# If SentenialAgent exposed proper BaseAgent subclass, register that instance.
try:
    # If SentenialAgent subclasses BaseAgent:
    o.register_agent("agent001", agent_wrapper)
    o.start_agent("agent001")
    print("Agent started; running for 10s")
    time.sleep(10)
    print("Stopping...")
    o.stop_all()
except Exception as e:
    print("Orchestrator run failed:", e)
PY
