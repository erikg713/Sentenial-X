from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from model_manager import ModelManager

app = FastAPI(
    title="Sentenial-X Inference API",
    description="Endpoints for compliance, vulnerability, pentest, intrusion, and HTTP analysis",
    version="0.1.0"
)

mm = ModelManager()

class AuditRequest(BaseModel):
    prompt: str
    http_payload: str = None

class AuditResponse(BaseModel):
    result: str

@app.post("/analyze", response_model=AuditResponse)
async def analyze(req: AuditRequest):
    try:
        out = mm.generate(req.prompt, req.http_payload)
        return AuditResponse(result=out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.api_server:app", host="0.0.0.0", port=8000, reload=False)
