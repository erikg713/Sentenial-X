"""
Sentenial-X AI Core: Orchestrator
---------------------------------
Coordinates AI workflows for Cortex, WormGPT, and other modules.
Handles task queuing, model execution, and result aggregation.

Author: Sentenial-X Development Team
"""

import logging
import uuid
from typing import Any, Dict, List, Optional
from collections import deque
from datetime import datetime
from api.utils.logger import init_logger
from ai_core.model_loader import ModelLoader

logger = init_logger("ai_core.orchestrator")


class AIOrchestrator:
    """
    Manages AI tasks, model execution, and workflow orchestration.
    """

    def __init__(self, max_history: int = 500):
        self.task_history: deque = deque(maxlen=max_history)
        self.model_loader = ModelLoader()
        logger.info("AIOrchestrator initialized with max history: %d", max_history)

    def create_task(
        self,
        model_name: str,
        payload: Dict[str, Any],
        task_type: str = "generic",
        priority: int = 0,
    ) -> str:
        """
        Create and queue a new AI task.
        """
        task_id = str(uuid.uuid4())
        task = {
            "task_id": task_id,
            "model_name": model_name,
            "payload": payload,
            "task_type": task_type,
            "priority": priority,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "queued",
            "result": None,
        }
        self.task_history.append(task)
        logger.info("Task created: %s", task_id)
        return task_id

    def execute_task(self, task_id: str) -> Dict[str, Any]:
        """
        Execute a task by loading the required model and processing the payload.
        """
        task = next((t for t in self.task_history if t["task_id"] == task_id), None)
        if not task:
            logger.error("Task not found: %s", task_id)
            raise ValueError(f"Task {task_id} not found")

        task["status"] = "running"
        model_name = task["model_name"]
        payload = task["payload"]

        logger.info("Executing task %s using model %s", task_id, model_name)

        try:
            model = self.model_loader.load_model(model_name)
            # Placeholder: replace with actual model inference
            result = self._mock_execute(model, payload)
            task["status"] = "completed"
            task["result"] = result
            logger.info("Task %s completed", task_id)
        except Exception as e:
            task["status"] = "failed"
            task["result"] = {"error": str(e)}
            logger.exception("Task %s failed: %s", task_id, e)

        return task

    def _mock_execute(self, model: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mock execution function. Replace with real AI inference.
        """
        logger.debug("Mock executing model %s with payload: %s", model["model_path"], payload)
        return {
            "model": model["model_path"],
            "payload": payload,
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
        }

    def list_tasks(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List tasks, optionally filtered by status.
        """
        if status_filter:
            return [t for t in self.task_history if t["status"] == status_filter]
        return list(self.task_history)

    def clear_tasks(self):
        """
        Clear all tasks from history.
        """
        logger.warning("Clearing all %d tasks", len(self.task_history))
        self.task_history.clear()


# ------------------------
# CLI / Test Example
# ------------------------
if __name__ == "__main__":
    orchestrator = AIOrchestrator()

    task_id = orchestrator.create_task(
        model_name="wormgpt",
        payload={"prompt": "Simulate attack vector"},
        task_type="emulation",
    )
    print("Created task:", task_id)

    result = orchestrator.execute_task(task_id)
    print("Execution result:", result)

    print("All tasks:", orchestrator.list_tasks())
