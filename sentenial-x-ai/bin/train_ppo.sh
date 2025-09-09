#!/usr/bin/env bash
# Train PPO using training/train_ppo.py
set -euo pipefail

# Defaults (can override with env or CLI args)
N_ENVS="${N_ENVS:-4}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-250000}"
CHUNK="${CHUNK:-50000}"
TB_LOG="${TB_LOG:-runs/ppo_main}"
SAVE_PATH="${SAVE_PATH:-checkpoints/ppo_main.zip}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50000}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

export PYTHONUNBUFFERED=1

echo "[+] Activating venv if present..."
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "[+] Starting PPO training"
python training/train_ppo.py \
  --n-envs "${N_ENVS}" \
  --policy "MlpPolicy" \
  --total-timesteps "${TOTAL_TIMESTEPS}" \
  --chunk "${CHUNK}" \
  --tb-log "${TB_LOG}" \
  --save-path "${SAVE_PATH}" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
  --log-level "${LOG_LEVEL}"
