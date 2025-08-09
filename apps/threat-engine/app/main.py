# threat-engine/app/main.py
from fastapi import FastAPI
from .sample_data import SAMPLE_ALERTS
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Mock Threat Engine", version="0.1.0")

class Alert(BaseModel):
    id: str
    severity: str
    title: str
    description: str

@app.get("/v1/alerts", response_model=List[Alert])
async def list_alerts(limit: int = 25):
    return SAMPLE_ALERTS[:limit]

@app.get("/v1/alerts/{alert_id}", response_model=Alert)
async def get_alert(alert_id: str):
    for a in SAMPLE_ALERTS:
        if a["id"] == alert_id:
            return a
    return {}