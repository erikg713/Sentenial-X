from pydantic import BaseModel

class AnalyzeRequest(BaseModel):
    input: str

@router.post("/analyze")
async def analyze_payload(payload: AnalyzeRequest):
    normalized = normalize_input(payload.input)
    signature_threat = detect_signatures(normalized)
    embedding_result = analyze_with_embeddings(normalized)

    return {
        "normalized_input": normalized,
        "signature_threat": signature_threat,
        "embedding_result": embedding_result
    }
