# cli/telemetry.py
import psutil
import asyncio
from datetime import datetime
from .logger import setup_logger
from .memory import run_async, enqueue_command
from .config import AGENT_ID

logger = setup_logger("telemetry")

async def collect_system_metrics():
    """
    Collect CPU, Memory, Disk, and Network usage metrics periodically.
    """
    while True:
        metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "net_io": psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        logger.info(f"Telemetry collected: {metrics}")
        await enqueue_command(AGENT_ID, "telemetry", metrics)
        await asyncio.sleep(10)
