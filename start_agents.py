# sentenial_x/start_agents.py

import asyncio
import logging
from agents.manager import AgentManager
from agents.endpoint_agent import EndpointAgent
from agents.sentenial_x_ai_bot import SentenialXAI
from agents.retaliation_bot import RetaliationBot
from agents.config import get_config, logger as config_logger

# ==============================
# Logging
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("SentenialX.AgentOrchestrator")

# ==============================
# Initialize Agent Manager
# ==============================
manager = AgentManager()

# Load centralized config
agent_config = get_config()

# ==============================
# Register Agents
# ==============================
manager.register_agent("endpoint_agent", EndpointAgent, config=agent_config)
manager.register_agent("ai_agent", SentenialXAI, config=agent_config)
manager.register_agent("retaliation_bot", RetaliationBot, config=agent_config)

# ==============================
# Async Orchestrator Loop
# ==============================
async def main():
    try:
        logger.info("[Orchestrator] Starting all agents...")
        await manager.start_all()

        logger.info("[Orchestrator] Agents are now running. Press Ctrl+C to stop.")

        # Continuous monitoring loop
        while True:
            agent_status = manager.list_agents()
            for name, is_running in agent_status.items():
                logger.info(f"[Orchestrator] {name} running: {is_running}")
            await asyncio.sleep(agent_config.get("heartbeat_interval", 10))

    except KeyboardInterrupt:
        logger.info("[Orchestrator] Keyboard interrupt received. Stopping all agents...")
        await manager.stop_all()
    except Exception as e:
        logger.exception(f"[Orchestrator] Critical error: {e}")
        await manager.stop_all()
    finally:
        logger.info("[Orchestrator] Shutdown complete.")

# ==============================
# Run Orchestrator
# ==============================
if __name__ == "__main__":
    asyncio.run(main())
