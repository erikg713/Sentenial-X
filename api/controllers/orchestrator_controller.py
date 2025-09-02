# api/controllers/orchestrator_controller.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from orchestrator.orchestrator import Orchestrator
from typing import List

router = APIRouter(prefix="/orchestrator", tags=["orchestrator"])

# --- Pydantic Models for Request/Response ---

class OrchestrationTask(BaseModel):
    name: str
    modules: List[str]
    target: str

class OrchestrationResponse(BaseModel):
    task_id: str
    status: str
    message: str


# --- Orchestrator Instance ---
orchestrator = Orchestrator()


# --- API Endpoints ---

@router.post("/start", response_model=OrchestrationResponse)
async def start_task(task: OrchestrationTask):
    """
    Start a new orchestration task across selected modules.
    """
    try:
        task_id = orchestrator.start_task(
            name=task.name,
            modules=task.modules,
            target=task.target
        )
        return OrchestrationResponse(
            task_id=task_id,
            status="running",
            message=f"Task '{task.name}' started successfully."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{task_id}", response_model=OrchestrationResponse)
async def check_status(task_id: str):
    """
    Check the status of a specific orchestration task.
    """
    try:
        status = orchestrator.get_status(task_id)
        return OrchestrationResponse(
            task_id=task_id,
            status=status,
            message=f"Status for task {task_id}: {status}"
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop/{task_id}", response_model=OrchestrationResponse)
async def stop_task(task_id: str):
    """
    Stop a running orchestration task.
    """
    try:
        orchestrator.stop_task(task_id)
        return OrchestrationResponse(
            task_id=task_id,
            status="stopped",
            message=f"Task {task_id} stopped successfully."
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_tasks():
    """
    List all active orchestration tasks.
    """
    try:
        tasks = orchestrator.list_tasks()
        return {"active_tasks": tasks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
