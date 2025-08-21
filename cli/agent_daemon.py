#!/usr/bin/env python3
import os
import json
import asyncio
import logging
import random
from pathlib import Path
from datetime import datetime
import subprocess

from cli.config import AGENT_ID, HEARTBEAT_INTERVAL, COMMAND_QUEUE, LOG_FILE
from cli.logger import setup_logger
from cli.memory import enqueue_command, run_async

# -------------------------
# Logging Setup
# -------------------------
logger = setup_logger("sentenial-agent")
os.makedirs(LOG_FILE.parent, exist_ok=True)

# -------------------------
# Heartbeat
# -------------------------
async def send_heartbeat():
    while True:
        heartbeat = {
            "agent": AGENT_ID,
            "status": "online",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        logger.info(f"Heartbeat sent: {heartbeat}")
        await enqueue_command(AGENT_ID, "heartbeat", heartbeat)
        await asyncio.sleep(HEARTBEAT_INTERVAL)

# -------------------------
# Command Listener
# -------------------------
async def listen_for_commands():
    os.makedirs(COMMAND_QUEUE.parent, exist_ok=True)
    while True:
        if COMMAND_QUEUE.exists():
            try:
                with COMMAND_QUEUE.open("r") as f:
                    commands = json.load(f)
                if AGENT_ID in commands:
                    for cmd in commands[AGENT_ID]:
                        logger.info(f"Executing command: {cmd}")
                        # Execute CLI command safely
                        try:
                            result = subprocess.run(
                                ["./sentenial_cli_full.py"] + cmd.split(),
                                capture_output=True,
                                text=True,
                                check=True
                            )
                            output = result.stdout.strip()
                            logger.info(f"Command output: {output}")
                            await enqueue_command(AGENT_ID, cmd, {"output": output})
                        except subprocess.CalledProcessError as e:
                            logger.error(f"Command failed: {e.stderr}")
                            await enqueue_command(AGENT_ID, cmd, {"error": e.stderr})
                    # Clear executed commands
                    del commands[AGENT_ID]
                    with COMMAND_QUEUE.open("w") as f:
                        json.dump(commands, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to process commands: {e}")
        await asyncio.sleep(5)  # Poll interval

# -------------------------
# WormGPT Detector Simulation
# -------------------------
async def run_wormgpt_detector_periodic():
    while True:
        # Example adversarial prompts
        prompt = random.choice([
            "Simulate ransomware",
            "Phishing email generation",
            "AI malware attack"
        ])
        temperature = 0.7
        cmd = f"wormgpt-detector -p \"{prompt}\" -t {temperature}"
        logger.info(f"Running WormGPT detector: {prompt}")
        try:
            result = subprocess.run(
                ["./sentenial_cli_full.py"] + cmd.split(),
                capture_output=True,
                text=True,
                check=True
            )
            output = result.stdout.strip()
            logger.info(f"WormGPT results: {output}")
            await enqueue_command(AGENT_ID, "wormgpt_detector", {"prompt": prompt, "output": output})
        except subprocess.CalledProcessError as e:
            logger.error(f"WormGPT detector failed: {e.stderr}")
            await enqueue_command(AGENT_ID, "wormgpt_detector", {"prompt": prompt, "error": e.stderr})
        await asyncio.sleep(30)  # Run every 30 seconds

# -------------------------
# Telemetry Streaming Simulation
# -------------------------
async def stream_telemetry():
    while True:
        telemetry = {
            "cpu": random.randint(5, 80),
            "memory": random.randint(10, 70),
            "network": random.randint(0, 100),
            "timestamp": datetime.utcnow().isoformat()
        }
        logger.info(f"Telemetry: {telemetry}")
        await enqueue_command(AGENT_ID, "telemetry", telemetry)
        await asyncio.sleep(10)

# -------------------------
# Main Daemon
# -------------------------
async def main():
    logger.info(f"Sentenial-X Agent '{AGENT_ID}' starting...")
    await asyncio.gather(
        send_heartbeat(),
        listen_for_commands(),
        run_wormgpt_detector_periodic(),
        stream_telemetry()
    )

if __name__ == "__main__":
    asyncio.run(main())            result = subprocess.run(
                ["./sentenial_cli_full.py"] + cmd.split(),
                capture_output=True,
                text=True,
                check=True
            )
            output = result.stdout.strip()
            logger.info(f"WormGPT results: {output}")
            await enqueue_command(AGENT_ID, "wormgpt_detector", {"prompt": prompt, "output": output})
        except subprocess.CalledProcessError as e:
            logger.error(f"WormGPT detector failed: {e.stderr}")
            await enqueue_command(AGENT_ID, "wormgpt_detector", {"prompt": prompt, "error": e.stderr})
        await asyncio.sleep(30)  # Run every 30 seconds

# -------------------------
# Telemetry Streaming Simulation
# -------------------------
async def stream_telemetry():
    while True:
        telemetry = {
            "cpu": random.randint(5, 80),
            "memory": random.randint(10, 70),
            "network": random.randint(0, 100),
            "timestamp": datetime.utcnow().isoformat()
        }
        logger.info(f"Telemetry: {telemetry}")
        await enqueue_command(AGENT_ID, "telemetry", telemetry)
        await asyncio.sleep(10)

# -------------------------
# Main Daemon
# -------------------------
async def main():
    logger.info(f"Sentenial-X Agent '{AGENT_ID}' starting...")
    await asyncio.gather(
        send_heartbeat(),
        listen_for_commands(),
        run_wormgpt_detector_periodic(),
        stream_telemetry()
    )

if __name__ == "__main__":
    asyncio.run(main())
