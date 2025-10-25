from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from .dependencies import get_current_user, get_db_session
from .database import User  # Assuming you have a User model
from .config import JWT_SECRET_KEY, JWT_ALGORITHM, JWT_ACCESS_TOKEN_EXPIRE_MINUTES
from jose import jwt
from datetime import datetime, timedelta
import httpx
from pydantic import BaseModel

router = APIRouter()

class TextInput(BaseModel):
    text: str
    event_id: str

@router.post("/cortex/analyze")
async def proxy_to_cortex(input_data: TextInput):
    async with httpx.AsyncClient() as client:
        response = await client.post("http://cortex:8002/analyze", json=input_data.dict())
        return response.json()

@router.get("/threats")
async with httpx.AsyncClient() as client:
        response = await client.get("http://threat-engine:8001/threats")
        return response.json()
    
router = APIRouter()


def create_access_token(subject: str, expires_delta: timedelta | None = None) -> str:
    from .config import JWT_SECRET_KEY, JWT_ALGORITHM, JWT_ACCESS_TOKEN_EXPIRE_MINUTES

    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode = {"sub": subject, "exp": expire}
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


@router.post("/auth/token", tags=["Authentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate user and return JWT token.
    Replace dummy logic with real user validation.
    """
    username = form_data.username
    password = form_data.password

    # TODO: Replace with DB verification logic
    if username != "admin" or password != "secret":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(username)
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/users/me", tags=["Users"])
async def read_current_user(current_user: str = Depends(get_current_user)):
    """
    Returns information about the currently authenticated user.
    """
    # Replace with DB query to fetch user info if needed
    return {"username": current_user}


@router.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "ok", "service": "api-gateway"}
