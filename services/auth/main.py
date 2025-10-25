from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .auth_handler import generate_token, validate_token

app = FastAPI(title="Auth Service")

class Login(BaseModel):
    username: str
    password: str

@app.post("/token")
async def login(login: Login):
    if login.username == "operator" and login.password == "pass":  # Mock
        return {"access_token": generate_token(login.username)}
    raise HTTPException(401, "Invalid credentials")

@app.get("/validate")
async def validate(token: str):
    return validate_token(token)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
