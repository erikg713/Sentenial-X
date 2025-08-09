# app/routers/auth_routes.py
from fastapi import APIRouter, HTTPException, status
from ..models import Token
from ..auth import create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/token", response_model=Token)
async def token(username: str, password: str):
    """
    Demo token endpoint.
    In production authenticate against your identity provider (OIDC/SAML/LDAP).
    """
    # demo: accept any non-empty username/password
    if not username or not password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing credentials")

    # mock roles/scopes mapping
    scopes = ["read:health"]
    if username == "admin":
        scopes += ["admin"]

    token = create_access_token(subject=username, scopes=scopes)
    return Token(access_token=token)