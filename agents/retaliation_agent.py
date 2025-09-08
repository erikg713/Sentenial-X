# agents/retaliation_bot.py

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from agents.base_agent import BaseAgent
from agents.config import AgentConfig
from core.engine.alert_dispatcher import AlertDispatcher
from core.engine.incident_logger import IncidentLogger
from core.engine.network_watcher import NetworkWatcher
from core.engine.process_inspector import ProcessInspector

logger = logging.getLogger("RetaliationBot")
logger.setLevel(logging.INFO)


class RetaliationBot(BaseAgent):
    """
    RetaliationBot listens for high-severity incidents and executes
    automated, controlled countermeasures in response to active threats.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(agent_id="retaliation_bot", config=config or AgentConfig())
        self.alert_dispatcher = AlertDispatcher()
        self.incident_logger = IncidentLogger()
        self.network_watcher = NetworkWatcher()
        self.process_inspector = ProcessInspector()
        self.countermeasures_enabled = True

    async def run(self) -> None:
        """Main loop that continuously monitors and responds to incidents."""
        logger.info("[RetaliationBot] Retaliation agent initialized and running...")
        while True:
            try:
                alert = await self.alert_dispatcher.get_next_alert(timeout=5)
                if alert:
                    await self.handle_alert(alert)
            except asyncio.TimeoutError:
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"[RetaliationBot] Error in run loop: {e}")

    async def handle_alert(self, alert: Dict[str, Any]) -> None:
        """
        Process an incoming alert and trigger countermeasures if needed.
        """
        severity = alert.get("severity", "low").lower()
        source_ip = alert.get("source_ip", "unknown")
        process_id = alert.get("process_id")

        logger.info(f"[RetaliationBot] Received alert: {alert}")

        if severity in {"high", "critical"}:
            logger.warning(f"[RetaliationBot] High severity threat detected from {source_ip}")
            await self.take_countermeasures(source_ip, process_id, alert)

    async def take_countermeasures(self, source_ip: str, process_id: Optional[int], alert: Dict[str, Any]) -> None:
        """
        Execute layered countermeasures: isolate process, block IP, deploy deception.
        """
        timestamp = datetime.utcnow().isoformat()
        cm_log = {"time": timestamp, "source_ip": source_ip, "process_id": process_id, "action": []}

        if not self.countermeasures_enabled:
            logger.warning("[RetaliationBot] Countermeasures disabled. Logging only.")
            await self.incident_logger.log_incident(alert)
            return

        # Step 1: Kill or isolate malicious process
        if process_id:
            try:
                success = self.process_inspector.terminate_process(process_id)
                if success:
                    logger.info(f"[RetaliationBot] Terminated malicious process {process_id}")
                    cm_log["action"].append(f"Terminated process {process_id}")
                else:
                    cm_log["action"].append(f"Failed to terminate process {process_id}")
            except Exception as e:
                cm_log["action"].append(f"Error terminating process {process_id}: {e}")

        # Step 2: Block hostile IP address
        if source_ip and source_ip != "unknown":
            try:
                success = self.network_watcher.block_ip(source_ip)
                if success:
                    logger.info(f"[RetaliationBot] Blocked hostile IP {source_ip}")
                    cm_log["action"].append(f"Blocked IP {source_ip}")
                else:
                    cm_log["action"].append(f"Failed to block IP {source_ip}")
            except Exception as e:
                cm_log["action"].append(f"Error blocking IP {source_ip}: {e}")

        # Step 3: Deploy deception (honeypot redirection)
        try:
            deception_result = await self.deploy_deception(source_ip)
            cm_log["action"].append(deception_result)
        except Exception as e:
            cm_log["action"].append(f"Error deploying deception: {e}")

        # Log retaliation actions
        await self.incident_logger.log_countermeasure(cm_log)

    async def deploy_deception(self, attacker_ip: str) -> str:
        """
        Redirect attacker traffic into a deception environment (honeypot).
        """
        await asyncio.sleep(0.5)  # simulate latency
        logger.info(f"[RetaliationBot] Redirected attacker {attacker_ip} into deception environment.")
        return f"Redirected attacker {attacker_ip} into honeypot"

    def enable_countermeasures(self) -> None:
        self.countermeasures_enabled = True
        logger.info("[RetaliationBot] Countermeasures ENABLED")

    def disable_countermeasures(self) -> None:
        self.countermeasures_enabled = False
        logger.info("[RetaliationBot] Countermeasures DISABLED")
