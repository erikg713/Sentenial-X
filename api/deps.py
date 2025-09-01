import os
import time
import hashlib
from typing import Optional
from fastapi import Header, HTTPException, Request, Depends

# Simple in-memory token bucket per client
_BUCKETS = {}
_RATE = int(os.getenv("RATE_LIMIT_RPS", "10"))
_BURST = int(os.getenv("RATE_LIMIT_BURST", "20"))
_API_KEY = os.getenv("API_KEY", "dev-api-key")  # set in .env for prod

def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    if _API_KEY and x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

def _bucket_key(request: Request) -> str:
    ip = request.client.host if request.client else "unknown"
    return hashlib.sha256(ip.encode()).hexdigest()

async def rate_limit(request: Request):
    now = time.time()
    key = _bucket_key(request)
    tokens, last = _BUCKETS.get(key, (_BURST, now))
    # refill
    tokens = min(_BURST, tokens + (now - last) * _RATE)
    if tokens < 1:
        raise HTTPException(status_code=429, detail="Too Many Requests")
    _BUCKETS[key] = (tokens - 1, now)

def secure_dep(request: Request, _=Depends(rate_limit), __=Depends(require_api_key)):
    return True
