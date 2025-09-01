import time
from typing import AsyncGenerator

async def aiter(gen):
    """Turn sync iterable into async generator (helper)."""
    for item in gen:
        yield item

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
