import asyncio
import websockets
import json

class WebSocketSignalConsumer:
    def __init__(self, uri="ws://localhost:8765"):
        self.uri = uri

    async def consume(self, callback):
        async with websockets.connect(self.uri) as websocket:
            while True:
                raw = await websocket.recv()
                signal = json.loads(raw)
                await callback(signal)
