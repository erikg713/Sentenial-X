# cli/cortex.py
import asyncio
import json
from datetime import datetime
from .logger import setup_logger
from .memory import enqueue_command, run_async
from .config import AGENT_ID

logger = setup_logger("cortex")

async def analyze_text_threats(texts: list):
    """
    Mock NLP threat analysis: classify text inputs and log results.
    """
    for t in texts:
        threat_level = "high" if "malware" in t.lower() else "low"
        result = {"text": t, "threat_level": threat_level, "timestamp": datetime.utcnow().isoformat()}
        logger.info(f"Cortex analyzed: {json.dumps(result)}")
        await enqueue_command(AGENT_ID, "cortex_analysis", result)
