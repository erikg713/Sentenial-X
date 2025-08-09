# app/deps.py
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from .auth import decode_token
from .models import TokenPayload
from .auth import decode_token, introspect_token_remote
import asyncio

async def require_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> TokenPayload:
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(...)
    token = credentials.credentials
    # try local decode
    try:
        return decode_token(token)
    except Exception:
        # fallback to remote introspection if configured
        if getattr(settings, "OIDC_INTROSPECTION_URL", None):
            try:
                data = await introspect_token_remote(token)
                # introspection response must include active=true and sub, scope
                if not data.get("active"):
                    raise HTTPException(status_code=401, detail="Invalid token (introspection)")
                scopes = data.get("scope", "")
                scopes_list = scopes.split() if isinstance(scopes, str) else data.get("scopes", [])
                return TokenPayload(sub=data.get("sub", "unknown"), scopes=scopes_list)
            except HTTPException:
                raise
            except Exception:
                raise HTTPException(status_code=401, detail="Token introspection failed")
        raise HTTPException(status_code=401, detail="Invalid or expired token")
bearer_scheme = HTTPBearer(auto_error=False)


def require_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> TokenPayload:
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    try:
        payload = decode_token(token)
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    return payload


def require_scope(scope: str):
    def _require(payload: TokenPayload = Depends(require_token)):
        if scope not in (payload.scopes or []):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient scope")
        return payload

    return _require