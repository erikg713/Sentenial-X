# routes/orchestrator.py

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
import asyncio
import logging
from fastapi import APIRouter, Depends, HTTPException
from ..models import OrchestratorRequest, OrchestratorResponse
from ..deps import secure_dep

router = APIRouter(prefix="/orchestrator", tags=["orchestrator"])

@router.post("/execute", response_model=OrchestratorResponse)
async def execute(req: OrchestratorRequest, _=Depends(secure_dep)):
    try:
        from cli.orchestrator import Orchestrator
    except Exception as e:
        raise HTTPException(500, f"Module import failed: {e}")

    orch = Orchestrator()
    result = await orch.execute(action=req.action, params=req.params)
    return OrchestratorResponse(action=req.action, status=result.get("status", "ok"), result=result)
# Import orchestrator services
from orchestrator.core_manager import CoreOrchestrator
from orchestrator.job_manager import JobManager
from orchestrator.state_manager import OrchestratorState

# Router
router = APIRouter(prefix="/orchestrator", tags=["Orchestrator"])

# Managers
orchestrator = CoreOrchestrator()
job_manager = JobManager()
state_manager = OrchestratorState()

logger = logging.getLogger("sentenial.orchestrator")


@router.get("/status")
async def get_orchestrator_status() -> Dict[str, Any]:
    """
    Return orchestrator global status and system state.
    """
    try:
        status_data = {
            "uptime": state_manager.get_uptime(),
            "active_jobs": job_manager.list_jobs(),
            "system_health": orchestrator.system_health(),
            "last_activity": state_manager.get_last_activity(),
        }
        return JSONResponse(content=status_data)
    except Exception as e:
        logger.exception("Failed to fetch orchestrator status")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dispatch")
async def dispatch_task(task: Dict[str, Any]):
    """
    Dispatch a task through the orchestrator.
    Example payload:
    {
        "module": "exploit",
        "action": "launch",
        "params": { "target": "192.168.1.5", "exploit": "ms17_010" }
    }
    """
    try:
        job_id = await orchestrator.dispatch_task(task)
        state_manager.update_last_activity()
        return {"message": "Task dispatched", "job_id": job_id}
    except Exception as e:
        logger.exception("Task dispatch failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Retrieve status of a specific job.
    """
    try:
        job_status = job_manager.get_job_status(job_id)
        if not job_status:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"job_id": job_id, "status": job_status}
    except Exception as e:
        logger.exception(f"Error retrieving job {job_id}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a running job if possible.
    """
    try:
        result = job_manager.cancel_job(job_id)
        if not result:
            raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")
        return {"message": f"Job {job_id} cancelled"}
    except Exception as e:
        logger.exception(f"Error cancelling job {job_id}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs")
async def get_logs(limit: int = Query(100, description="Number of log lines to fetch")) -> Dict[str, Any]:
    """
    Retrieve orchestrator logs for debugging or audit.
    """
    try:
        logs = state_manager.get_logs(limit=limit)
        return {"logs": logs}
    except Exception as e:
        logger.exception("Failed to fetch orchestrator logs")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset")
async def reset_orchestrator():
    """
    Reset orchestrator state (dangerous).
    """
    try:
        orchestrator.reset()
        state_manager.reset()
        job_manager.reset()
        return {"message": "Orchestrator reset successfully"}
    except Exception as e:
        logger.exception("Failed to reset orchestrator")
        raise HTTPException(status_code=500, detail=str(e)) 
