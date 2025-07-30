# sentenial_x/core/cortex/server.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .model_loader import CyberIntentModel

app = FastAPI(title="Sentenial-X Cortex NLP API")

model = CyberIntentModel()

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    intent_label: str

@app.post("/predict", response_model=PredictResponse)
async def predict_intent(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    label = model.predict(request.text)
    return PredictResponse(intent_label=label)

@app.get("/")
async def root():
    return {"message": "Sentenial-X Cortex NLP API running"}

# Run with:
# uvicorn sentenial_x.core.cortex.server:app --host 0.0.0.0 --port 8080

