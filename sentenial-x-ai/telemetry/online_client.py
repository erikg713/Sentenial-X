"""
Sentenial-X AI Telemetry Online Client
--------------------------------------

Handles real-time streaming of telemetry (threat events, system metrics, logs)
to a central telemetry server for monitoring and analysis.
"""

import asyncio
import json
import logging
import websockets
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger("sentenial.telemetry")


class TelemetryClient:
    def __init__(self, server_url: str = "ws://localhost:8765", client_id: str = "agent-001"):
        self.server_url = server_url
        self.client_id = client_id
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False

    async def connect(self):
        """Establish WebSocket connection to telemetry server."""
        try:
            self.ws = await websockets.connect(self.server_url)
            self.connected = True
            logger.info(f"[Telemetry] Connected to {self.server_url} as {self.client_id}")
            await self.send_heartbeat()
        except Exception as e:
            logger.error(f"[Telemetry] Failed to connect: {e}")
            self.connected = False

    async def send_heartbeat(self):
        """Send heartbeat to confirm liveness."""
        payload = {
            "type": "heartbeat",
            "client_id": self.client_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.send(payload)

    async def send(self, data: Dict[str, Any]):
        """Send structured telemetry data."""
        if not self.connected or self.ws is None:
            logger.warning("[Telemetry] Not connected, skipping send")
            return

        try:
            payload = json.dumps(data)
            await self.ws.send(payload)
            logger.debug(f"[Telemetry] Sent: {payload}")
        except Exception as e:
            logger.error(f"[Telemetry] Failed to send: {e}")
            self.connected = False

    async def send_event(self, event_type: str, details: Dict[str, Any]):
        """Send a specific event type."""
        payload = {
            "type": "event",
            "event_type": event_type,
            "client_id": self.client_id,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details,
        }
        await self.send(payload)

    async def send_metric(self, metric_name: str, value: Any):
        """Send system metric (e.g., CPU, memory, threat counts)."""
        payload = {
            "type": "metric",
            "metric": metric_name,
            "value": value,
            "client_id": self.client_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.send(payload)

    async def close(self):
        """Gracefully close connection."""
        if self.ws:
            await self.ws.close()
            logger.info("[Telemetry] Connection closed")
        self.connected = False


# Example standalone test
async def main():
    client = TelemetryClient("ws://localhost:8765", client_id="sentenial-demo")
    await client.connect()
    await client.send_event("threat_detected", {"ip": "192.168.1.10", "severity": "high"})
    await client.send_metric("cpu_usage", 74.3)
    await asyncio.sleep(1)
    await client.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(main())
