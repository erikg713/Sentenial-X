# apps/dashboard/ws_server.py
import asyncio
import json
import websockets
from live_dashboard_runner import DashboardOrchestrator

# Port for the WebSocket server
WS_PORT = 8765

class WebSocketDashboardServer:
    def __init__(self, orchestrator: DashboardOrchestrator):
        self.orchestrator = orchestrator
        self.clients = set()

    async def register(self, websocket):
        self.clients.add(websocket)
        print(f"[WS] Client connected: {websocket.remote_address}")

    async def unregister(self, websocket):
        self.clients.remove(websocket)
        print(f"[WS] Client disconnected: {websocket.remote_address}")

    async def send_dashboard(self):
        """
        Gather dashboard data and broadcast to all connected clients
        """
        data = self.orchestrator.home.render()  # Render returns dict
        if self.clients:
            message = json.dumps(data, default=str)
            await asyncio.gather(*[client.send(message) for client in self.clients])

    async def handler(self, websocket, path):
        await self.register(websocket)
        try:
            async for _ in websocket:  # Keep connection alive
                pass
        finally:
            await self.unregister(websocket)

    async def run(self, interval=5):
        """
        Start WebSocket server and broadcast dashboard updates
        """
        server = await websockets.serve(self.handler, "0.0.0.0", WS_PORT)
        print(f"[WS] Dashboard WebSocket running on ws://0.0.0.0:{WS_PORT}")
        while True:
            await self.orchestrator.run_services()
            await self.send_dashboard()
            await asyncio.sleep(interval)
        await server.wait_closed()

# CLI Entrypoint
if __name__ == "__main__":
    orchestrator = DashboardOrchestrator()
    ws_server = WebSocketDashboardServer(orchestrator)
    try:
        asyncio.run(ws_server.run(interval=5))
    except KeyboardInterrupt:
        print("\n[WebSocket Dashboard terminated by user]")
