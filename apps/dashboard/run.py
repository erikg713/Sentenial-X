# apps/dashboard/run.py
import asyncio
from apps.dashboard.main import run_dashboard
from apps.dashboard.api.ws_server import start_server

async def main():
    await asyncio.gather(
        run_dashboard(),
        start_server()
    )

if __name__ == "__main__":
    asyncio.run(main())
