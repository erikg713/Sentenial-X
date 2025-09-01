from fastapi import APIRouter, Depends, HTTPException
from ..models import WormGPTRequest, WormGPTResponse
from ..deps import secure_dep

router = APIRouter(prefix="/wormgpt", tags=["wormgpt"])

@router.post("/detect", response_model=WormGPTResponse)
async def detect(req: WormGPTRequest, _=Depends(secure_dep)):
    # Integrate cli.wormgpt
    try:
        from cli.wormgpt import WormGPT
    except Exception as e:
        raise HTTPException(500, f"Module import failed: {e}")

    detector = WormGPT()
    res = await detector.detect(prompt=req.prompt, temperature=req.temperature)
    return WormGPTResponse(**res)
