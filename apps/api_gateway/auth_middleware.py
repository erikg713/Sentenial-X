from fastapi import HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    if token != "valid-token":  # Mock auth
        raise HTTPException(status_code=401, detail="Invalid token")
    return {"username": "operator"}
