# app/models.py
from pydantic import BaseModel
from typing import List, Optional


class HealthResponse(BaseModel):
    service: str
    status: str
    version: str = "alpha"
    uptime_seconds: Optional[int] = None


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenPayload(BaseModel):
    sub: str
    scopes: List[str] = []
    exp: int | None = None