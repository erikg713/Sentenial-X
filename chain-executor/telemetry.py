import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger("Telemetry")
logging.basicConfig(level=logging.INFO)

class TelemetryEmitter:
    def __init__(self, emit_func: Optional[callable] = None):
        """
        Initializes the TelemetryEmitter.
        :param emit_func: Optional function to forward telemetry externally (e.g. WebSocket, Kafka).
        """
        self.emit_func = emit_func

    def emit(
        self,
        event_type: str,
        payload: Dict[str, Any],
        session_id: Optional[str] = None,
        step: Optional[str] = None,
    ):
        """
        Emits a structured telemetry event.
        :param event_type: Category or type of event (e.g. "step_start", "step_complete", "error")
        :param payload: Dictionary of relevant event data
        :param session_id: Optional session UUID or trace ID
        :param step: Optional name of the current step in execution
        """
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            "session_id": session_id,
            "step": step,
            "data": payload,
        }

        # Log locally
        logger.info(f"[TELEMETRY] {json.dumps(event)}")

        # Emit to external system if configured
        if self.emit_func:
            try:
                self.emit_func(event)
            except Exception as e:
                logger.warning(f"Failed to emit telemetry externally: {e}")

