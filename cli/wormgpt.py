#!/usr/bin/env python3
"""
cli/wormgpt.py

WormGPT-style adversarial AI prompt detector for Sentenial-X.
Designed for CLI usage via sentenial_cli_full.py.
"""

import asyncio
import json
from datetime import datetime
from memory import write_command  # adjust import to your actual memory/logging module
from logger import log_event  # optional logging helper
from wormgpt_core import analyze_prompt  # your core AI detection logic

async def run_wormgpt(prompt: str, temperature: float = 0.7) -> dict:
    """
    Run adversarial prompt detection and generate countermeasures.
    Args:
        prompt (str): The input text to analyze.
        temperature (float): Randomness factor for AI detection.
    Returns:
        dict: Detection results including risk, findings, countermeasures, timestamp.
    """
    # Ensure basic validation
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Prompt must be a non-empty string.")

    # Call your detection engine
    try:
        results = await analyze_prompt(prompt, temperature=temperature)
    except Exception as e:
        results = {"error": str(e), "action": "wormgpt-detector"}

    # Construct full result
    output = {
        "action": "wormgpt-detector",
        "prompt_risk": results.get("risk", "unknown"),
        "detections": results.get("detections", []),
        "countermeasures": results.get("countermeasures", []),
        "temperature": temperature,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    # Persist to memory/log
    try:
        write_command(action="wormgpt-detector", params={"prompt": prompt, "temperature": temperature}, result=output)
    except Exception as e:
        log_event(f"Failed to write WormGPT results to memory: {e}")

    return output

# CLI handler for sentenial_cli_full.py
async def handle_wormgpt(args):
    """
    Handler for CLI subparser: wormgpt-detector
    Example usage:
        ./sentenial_cli_full.py wormgpt-detector -p "example prompt" -t 0.8
    """
    result = await run_wormgpt(prompt=args.prompt, temperature=args.temperature)
    print(json.dumps(result, indent=2))
