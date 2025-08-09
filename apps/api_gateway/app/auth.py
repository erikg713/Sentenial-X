# app/auth.py
import time
from typing import List
from jose import jwt, JWTError
from datetime import datetime, timedelta
from .config import settings
from .models import TokenPayload

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