from fastapi import FastAPI, Request
from router import router
import requests
from sentenialx.models.artifacts import verify_artifact

def analyze_threat(log_data: str):
    if not verify_artifact("encoder"):  # Use encoder for embeddings
        raise ValueError("Encoder integrity failed!")
    
    # Call Cortex for NLP intent
    cortex_response = requests.post("http://api-gateway:8000/cortex/predict", json={"text": log_data}).json()
    intent = cortex_response["intent"]
    
    # Multi-modal fusion placeholder
    score = 0.8 if intent == "MALICIOUS" else 0.2  # Combine with rules/ML
    return {"threat_score": score, "intent": intent}
app = FastAPI(title="Threat Semantics Engine")
app.include_router(router)

@app.get("/ping")
def ping():
    return {"status": "ok"}
