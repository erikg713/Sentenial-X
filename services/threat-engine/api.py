from fastapi import FastAPI
from pydantic import BaseModel
from .classifier import ThreatClassifier
from .config import Config

app = FastAPI(title="Sentenial X - Threat Engine", version="1.0.0")

clf = ThreatClassifier()

class Payload(BaseModel):
    data: str

@app.post("/classify")
def classify(payload: Payload):
    result = clf.classify(payload.data)
    return {"result": result}
