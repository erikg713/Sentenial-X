# apps/dashboard/api/ws_server.py
import asyncio
import websockets
from apps.dashboard.pages.live_update import LiveUpdate
import json

async def handler(websocket, path):
    updater = LiveUpdate()
    while True:
        data = updater.loader.load_all()
        await websocket.send(json.dumps(data))
        await asyncio.sleep(3)

async def start_server():
    server = await websockets.serve(handler, "0.0.0.0", 8765)
    await server.wait_closed()
