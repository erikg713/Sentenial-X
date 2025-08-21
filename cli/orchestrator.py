# cli/orchestrator.py
import asyncio
import logging
from datetime import datetime
from .logger import setup_logger
from .memory import enqueue_command, run_async
from .config import AGENT_ID

logger = setup_logger("orchestrator")

async def orchestrate_command(action: str, params: dict = None):
    """
    Orchestrates multi-step agent commands.
    """
    logger.info(f"Orchestrator running action: {action} with params: {params}")
    # Example: trigger Cortex analysis
    if action == "analyze_texts" and params:
        from .cortex import analyze_text_threats
        await analyze_text_threats(params.get("texts", []))
    await enqueue_command(AGENT_ID, "orchestrator_action", {"action": action, "params": params, "timestamp": datetime.utcnow().isoformat()})
