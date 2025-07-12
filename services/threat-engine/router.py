from fastapi import APIRouter, Request
from analyzers.embedding_analyzer import analyze_with_embeddings
from analyzers.payload_normalizer import normalize_input

router = APIRouter()

@router.post("/analyze")
async def analyze_payload(request: Request):
    data = await request.json()
    raw_input = data.get("input", "")

    normalized = normalize_input(raw_input)
    result = analyze_with_embeddings(normalized)
    return {"result": result}
