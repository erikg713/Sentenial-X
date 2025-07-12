from fastapi import FastAPI, Request
from router import router

app = FastAPI(title="Threat Semantics Engine")
app.include_router(router)

@app.get("/ping")
def ping():
    return {"status": "ok"}
