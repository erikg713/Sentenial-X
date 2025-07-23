import asyncio
from datetime import datetime
from typing import Any, Callable, Coroutine, List
from aiohttp import web, ClientSession
from config import AGENT_ID, NETWORK_PORT, PEERS

class NetworkAdapter:
    def __init__(self):
        self._app = web.Application()
        self._app.add_routes([
            web.post('/message', self._on_receive),
            web.get('/health', self._on_health)
        ])
        self._receive_callback: Callable[[dict[str, Any]], Coroutine] | None = None

    def on_message(self, callback: Callable[[dict[str, Any]], Coroutine]) -> None:
        self._receive_callback = callback

    async def _on_receive(self, request: web.Request) -> web.Response:
        payload = await request.json()
        if self._receive_callback:
            await self._receive_callback(payload)
        return web.Response(status=204)

   
