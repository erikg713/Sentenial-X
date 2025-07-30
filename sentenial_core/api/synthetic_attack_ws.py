# sentenial_core/api/synthetic_attack_ws.py

import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from loguru import logger
from typing import List, Set

from sentenial_core.simulator.async_synthetic_attack_fuzzer import AsyncSyntheticAttackFuzzer

app = FastAPI()
fuzzer = AsyncSyntheticAttackFuzzer()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket client connected: {websocket.client}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket client disconnected: {websocket.client}")

    async def broadcast(self, message: dict):
        to_remove = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send WS message: {e}")
                to_remove.append(connection)
        for conn in to_remove:
            self.disconnect(conn)


manager = ConnectionManager()

@app.websocket("/ws/synthetic-attacks")
async def synthetic_attack_ws(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Stream events indefinitely until client disconnects
        async for event in fuzzer.stream_events(delay=1.0):
            await manager.broadcast(event)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)
