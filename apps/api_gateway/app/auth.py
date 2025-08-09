# app/auth.py
import time
from typing import List
from jose import jwt, JWTError
from datetime import datetime, timedelta
from .config import settings
from .models import TokenPayload
import httpx
from .config import settings

async def introspect_token_remote(token: str):
    if not settings.OIDC_INTROSPECTION_URL:
        raise ValueError("No introspection endpoint configured")
    auth = None
    if settings.OIDC_INTROSPECTION_CLIENT_ID and settings.OIDC_INTROSPECTION_CLIENT_SECRET:
        auth = (settings.OIDC_INTROSPECTION_CLIENT_ID, settings.OIDC_INTROSPECTION_CLIENT_SECRET)
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.post(
            settings.OIDC_INTROSPECTION_URL,
            data={"token": token},
            auth=auth,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()

def decode_token(token: str) -> TokenPayload:
    # first try local JWT decode
    try:
        payload = jwt.decode(token, SECRET, algorithms=[ALGORITHM])
        return TokenPayload(**payload)
    except JWTError:
        # fallback: if OIDC introspection configured, call it synchronously via httpx (make wrapper async earlier)
        if settings.OIDC_INTROSPECTION_URL:
            # note: require async call at higher level — for simplicity throw an instructive error
            raise RuntimeError("Token appears opaque. Use require_token_async which supports OIDC introspection.")
        raise ValueError("Invalid token")

ALGORITHM = settings.JWT_ALGORITHM
SECRET = settings.JWT_SECRET


def create_access_token(subject: str, scopes: List[str] = [], expires_in: int | None = None) -> str:
    now = int(time.time())
    exp = now + (expires_in or settings.ACCESS_TOKEN_EXPIRE_SECONDS)
    payload = {
        "sub": subject,
        "scopes": scopes,
        "iat": now,
        "exp": exp,
    }
    token = jwt.encode(payload, SECRET, algorithm=ALGORITHM)
    return token


def decode_token(token: str) -> TokenPayload:
    try:
        payload = jwt.decode(token, SECRET, algorithms=[ALGORITHM])
        return TokenPayload(**payload)
    except JWTError as e:
        raise ValueError("Invalid token") from e