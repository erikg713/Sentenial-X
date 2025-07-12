fromfrom fastapi import APIRouter, Request
from analyzers.embedding_analyzer import analyze_with_embeddings
from analyzers.payload_normalizer import normalize_input
from analyzers.signature_detector import detect_signatures

router = APIRouter()

@router.post("/analyze")
async def analyze_payload(request: Request):
    data = await request.json()
    raw_input = data.get("input", "")

    normalized = normalize_input(raw_input)
    signature_threat = detect_signatures(normalized)
    embedding_result = analyze_with_embeddings(normalized)

    return {
        "normalized_input": normalized,
        "signature_threat": signature_threat,
        "embedding_result": embedding_result
    }
