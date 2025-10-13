# services/cortex/server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentenialx.models.artifacts import get_artifact_path, verify_artifact
from transformers import pipeline, BertForSequenceClassification, BertTokenizer
import logging

# services/cortex/server.py (excerpt)
from peft import PeftModel
from sentenialx.models.artifacts import get_artifact_path, verify_artifact

MODEL_TYPE = "lora"
if not verify_artifact(MODEL_TYPE):
    raise RuntimeError("LoRA integrity failed!")
base_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
lora_path = get_artifact_path(MODEL_TYPE)
model = PeftModel.from_pretrained(base_model, lora_path.parent)

app = FastAPI(title="Cortex NLP API")
logging.basicConfig(level=logging.INFO)

class PredictRequest(BaseModel):
    text: str

# Load model at startup
MODEL_TYPE = "distill"  # From env or config
if not verify_artifact(MODEL_TYPE):
    raise RuntimeError("Model integrity failed!")
model_path = get_artifact_path(MODEL_TYPE)
tokenizer = BertTokenizer.from_pretrained(model_path.parent)
model = BertForSequenceClassification.from_pretrained(model_path.parent)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        result = classifier(request.text)
        # Fuse with Threat Engine (placeholder: forward to internal API)
        # requests.post("http://threat-engine:8001/score", json={"intent": result})
        return {"intent": result[0]['label'], "confidence": result[0]['score']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy"}
