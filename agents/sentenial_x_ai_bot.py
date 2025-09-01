# agents/sentenial_x_ai_bot.py

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import torch
from core.semantic_analyzer.models.transformer import TransformerEncoder
from core.semantic_analyzer.server import SemanticAnalyzerServer
from core.engine.alert_dispatcher import AlertDispatcher
from core.engine.incident_logger import IncidentLogger

from agents.base_agent import BaseAgent
from agents.config import AgentConfig

logger = logging.getLogger("SentenialX.AIBot")
logger.setLevel(logging.INFO)


class SentenialXAI(BaseAgent):
    """
    AI-powered agent for autonomous threat detection and response.
    Uses semantic analysis on telemetry events to classify anomalies
    and trigger alerts or countermeasures.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(agent_id="sentenial_x_ai", config=config or AgentConfig())
        self.alert_dispatcher = AlertDispatcher()
        self.incident_logger = IncidentLogger()
        self.model: Optional[TransformerEncoder] = None
        self.loop = asyncio.get_event_loop()
        self.running = True

    def setup(self):
        """Load AI model and prepare inference server"""
        logger.info("[SentenialXAI] Loading Transformer model...")
        try:
            self.model = TransformerEncoder(embedding_dim=self.config.get("embedding_dim", 128))
            model_path = self.config.get("model_path", "models/telemetry_encoder.pt")
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            logger.info(f"[SentenialXAI] Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"[SentenialXAI] Failed to load model: {e}")
            self.model = None

        self.semantic_server = SemanticAnalyzerServer(model=self.model)
        logger.info("[SentenialXAI] Semantic analyzer server initialized.")

    def execute(self):
        """Main loop to fetch telemetry events and analyze them"""
        if not self.running:
            return

        # Fetch events from alert dispatcher or event queue
        try:
            event = self.alert_dispatcher.get_next_alert(timeout=2)
            if event:
                self.loop.run_until_complete(self.analyze_event(event))
        except Exception as e:
            logger.error(f"[SentenialXAI] Error fetching event: {e}")

    async def analyze_event(self, event: Dict[str, Any]):
        """Analyze telemetry using AI model"""
        if not self.model:
            logger.warning("[SentenialXAI] No model loaded, skipping analysis.")
            return

        try:
            # Semantic scoring
            score = self.semantic_server.score_event(event)
            event["stealth_score"] = score

            # Log anomalies
            if score > 0.7:
                logger.warning(f"[SentenialXAI] High anomaly score detected: {score}")
                await self.raise_alert(event)
            else:
                logger.info(f"[SentenialXAI] Event score: {score}")

        except Exception as e:
            logger.error(f"[SentenialXAI] Analysis failed: {e}")

    async def raise_alert(self, event: Dict[str, Any]):
        """Send high-risk telemetry event to the alert dispatcher"""
        try:
            event_payload = {
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat(),
                "event": event,
                "severity": "high" if event["stealth_score"] > 0.85 else "medium",
            }
            await self.alert_dispatcher.dispatch_alert(event_payload)
            await self.incident_logger.log_incident(event_payload)
            logger.info(f"[SentenialXAI] Alert dispatched for event: {event_payload}")
        except Exception as e:
            logger.error(f"[SentenialXAI] Failed to raise alert: {e}")

    def teardown(self):
        """Cleanup resources"""
        self.running = False
        logger.info("[SentenialXAI] Agent shutting down, resources cleaned up.")
