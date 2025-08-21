# cli/wormgpt.py
import random
import subprocess
import asyncio
from datetime import datetime
from .logger import setup_logger
from .memory import enqueue_command, run_async
from .config import AGENT_ID

logger = setup_logger("wormgpt")

async def run_wormgpt_simulation():
    """
    Periodically simulate WormGPT-style threats.
    """
    prompts = [
        "Simulate ransomware",
        "Phishing campaign",
        "AI malware injection"
    ]
    while True:
        prompt = random.choice(prompts)
        cmd = f"wormgpt-detector -p \"{prompt}\" -t 0.7"
        logger.info(f"Running WormGPT simulation: {prompt}")
        try:
            result = subprocess.run(
                ["./sentenial_cli_full.py"] + cmd.split(),
                capture_output=True,
                text=True,
                check=True
            )
            output = result.stdout.strip()
            logger.info(f"WormGPT output: {output}")
            await enqueue_command(AGENT_ID, "wormgpt_simulation", {"prompt": prompt, "output": output})
        except subprocess.CalledProcessError as e:
            logger.error(f"WormGPT simulation failed: {e.stderr}")
            await enqueue_command(AGENT_ID, "wormgpt_simulation", {"prompt": prompt, "error": e.stderr})
        await asyncio.sleep(30)
