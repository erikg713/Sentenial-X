#!/bin/bash
echo "[+] Setting up Sentenial-X AI environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

echo "[+] Running training script..."
python training/train_agent.py
