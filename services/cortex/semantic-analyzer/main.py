from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from .analyzer import SemanticAnalyzer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sentenial-X Cortex Semantic Analyzer")

# Input model
class TextInput(BaseModel):
    text: str
    event_id: str

# Output model
class IntentResult(BaseModel):
    event_id: str
    intent: str
    confidence: float
    is_threat: bool

# Initialize analyzer
analyzer = SemanticAnalyzer(model_path="bert-base-uncased")  # Mock path, replace with actual

@app.post("/analyze", response_model=IntentResult)
async def analyze_text(input_data: TextInput):
    try:
        intent, confidence = analyzer.predict_intent(input_data.text)
        is_threat = confidence > 0.5 and intent in ["malicious", "phishing", "exploitation"]
        logger.info(f"Analyzed event {input_data.event_id}: intent={intent}, confidence={confidence}")
        return IntentResult(
            event_id=input_data.event_id,
            intent=intent,
            confidence=confidence,
            is_threat=is_threat
        )
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
