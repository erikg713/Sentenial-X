# app/rbac.py
import json
from typing import List, Optional
import redis.asyncio as redis
from .config import settings

REDIS_URL = getattr(settings, "REDIS_URL", "redis://redis:6379/0")
r = redis.from_url(REDIS_URL, decode_responses=True)


async def set_role(role_name: str, scopes: List[str]):
    key = f"role:{role_name}"
    await r.set(key, json.dumps(scopes))
    return True


async def get_role(role_name: str) -> Optional[List[str]]:
    key = f"role:{role_name}"
    v = await r.get(key)
    if not v:
        return None
    return json.loads(v)


async def delete_role(role_name: str):
    key = f"role:{role_name}"
    await r.delete(key)
    return True


async def list_roles():
    keys = await r.keys("role:*")
    results = {}
    for k in keys:
        role = k.split("role:")[1]
        results[role] = json.loads(await r.get(k))
    return results