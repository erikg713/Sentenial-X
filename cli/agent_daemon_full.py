#!/usr/bin/env python3
import asyncio
from cli.logger import setup_logger
from cli.memory import enqueue_command, run_async
from cli.config import AGENT_ID
from cli import telemetry, wormgpt, cortex, orchestrator, alerts

logger = setup_logger("sentenial-agent-full")

# -------------------------
# Heartbeat
# -------------------------
async def heartbeat_loop(interval: int = 10):
    while True:
        heartbeat = {
            "agent": AGENT_ID,
            "status": "online",
            "timestamp": asyncio.get_event_loop().time()
        }
        logger.info(f"Heartbeat sent: {heartbeat}")
        await enqueue_command(AGENT_ID, "heartbeat", heartbeat)
        await asyncio.sleep(interval)

# -------------------------
# Telemetry
# -------------------------
async def telemetry_loop():
    await telemetry.collect_system_metrics()

# -------------------------
# WormGPT Threat Simulation
# -------------------------
async def wormgpt_loop():
    await wormgpt.run_wormgpt_simulation()

# -------------------------
# Cortex NLP Analysis
# -------------------------
async def cortex_loop():
    # Example: periodic analysis of fake input
    sample_texts = [
        "Suspicious login detected",
        "User downloaded malware",
        "System running normally"
    ]
    while True:
        await cortex.analyze_text_threats(sample_texts)
        await asyncio.sleep(20)

# -------------------------
# Orchestrator
# -------------------------
async def orchestrator_loop():
    # Example: orchestrate multi-step command
    while True:
        action = "analyze_texts"
        params = {"texts": ["New suspicious activity detected", "Possible malware signature found"]}
        await orchestrator.orchestrate_command(action, params)
        await asyncio.sleep(25)

# -------------------------
# Alert Dispatcher (Simulated)
# -------------------------
async def alerts_loop():
    while True:
        # Dispatch random alerts for demonstration
        alert_type = "system_anomaly"
        severity = "medium"
        details = {"description": "Random simulated anomaly"}
        alerts.dispatch_alert(alert_type, severity, details)
        await asyncio.sleep(30)

# -------------------------
# Main Async Daemon
# -------------------------
async def main():
    logger.info(f"Sentenial-X Full Agent '{AGENT_ID}' starting...")
    await asyncio.gather(
        heartbeat_loop(),
        telemetry_loop(),
        wormgpt_loop(),
        cortex_loop(),
        orchestrator_loop(),
        alerts_loop()
    )

if __name__ == "__main__":
    asyncio.run(main())
