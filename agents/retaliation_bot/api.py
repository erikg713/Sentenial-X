from fastapi import FastAPI
from pydantic import BaseModel
from bot import RetaliationBot
from models import ThreatEvent
import asyncio

app = FastAPI()
bot = RetaliationBot()
bot.activate()

class ThreatInput(BaseModel):
    source_ip: str
    vector: str
    severity: int
    details: dict = {}

@app.post("/threat")
async def submit_threat(threat: ThreatInput):
    event = ThreatEvent(
        source_ip=threat.source_ip,
        vector=threat.vector,
        severity=threat.severity,
        details=threat.details
    )
    await bot.handle_event(event)
    return {"message": "Threat handled."}

@app.get("/status")
def status():
    return {"active": bot.is_active()}

@app.post("/activate")
def activate():
    bot.activate()
    return {"message": "Bot activated."}

@app.post("/deactivate")
def deactivate():
    bot.deactivate()
    return {"message": "Bot deactivated."}