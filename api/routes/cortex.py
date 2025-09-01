from fastapi import APIRouter, Depends, HTTPException
from ..models import CortexRequest
from ..deps import secure_dep

router = APIRouter(prefix="/cortex", tags=["cortex"])

@router.post("/analyze")
async def analyze(req: CortexRequest, _=Depends(secure_dep)):
    """
    Calls the CLI cortex analyzer; expects it to return structured findings.
    """
    try:
        from cli.cortex import Cortex
    except Exception as e:
        raise HTTPException(500, f"Module import failed: {e}")

    analyzer = Cortex()
    resp = await analyzer.analyze(source=req.source, filter_expr=req.filter or "")
    # Return pass-through JSON (already structured)
    return resp
