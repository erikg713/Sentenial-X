# ai_core/attack_simulator.py
"""
Robust, optimized attack simulation wrapper.

Improvements over original:
- Added input validation and sanitization.
- Explicit typing and better return structure (keeps backward compatibility).
- Timeout + retry logic for enqueue_task with exponential backoff.
- Structured logging for easier debugging.
- Clearer docstring and parameterization (complexity, timeout, retries).
"""
from typing import Any, Dict, Optional
import asyncio
import logging

from .predictive_model import enqueue_task, select_model
from .utils import preprocess_input

logger = logging.getLogger(__name__)


async def simulate_attack(
    prompt_text: str,
    complexity: str = "high",
    timeout: float = 30.0,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Run a multi-step attack simulation.

    This function:
    - validates and preprocesses the input,
    - selects an appropriate model for the requested complexity,
    - enqueues the preprocessing result to the predictive model subsystem with
      sensible timeout and retry behavior.

    Parameters:
    - prompt_text: raw text to simulate against (required).
    - complexity: desired complexity/strength of model ("low"|"medium"|"high").
    - timeout: per-attempt timeout in seconds for enqueue_task.
    - max_retries: number of retries on transient failures (0 = no retries).

    Returns:
    A dictionary with at least:
    - "model_used": str name of the model selected
    - "simulation_output": raw output from enqueue_task (may be None on failure)
    - "status": "success" | "timeout" | "failed"
    - "error": optional string message on failure

    Notes:
    - The return shape remains backward-compatible by including the original keys
      "model_used" and "simulation_output". Callers can also inspect "status"
      / "error" for more details.
    """
    # Basic validation
    if not isinstance(prompt_text, str):
        raise TypeError("prompt_text must be a string")

    prompt = prompt_text.strip()
    if not prompt:
        raise ValueError("prompt_text cannot be empty or whitespace")

    # Safety/size limits to avoid overloading downstream systems
    MAX_INPUT_CHARS = 10000
    if len(prompt) > MAX_INPUT_CHARS:
        logger.warning("Input truncated to %d characters", MAX_INPUT_CHARS)
        prompt = prompt[:MAX_INPUT_CHARS]

    # Preprocess input (may raise — let caller handle unexpected fatal errors)
    try:
        processed = preprocess_input(prompt)
    except Exception as exc:
        logger.exception("preprocess_input failed")
        return {
            "model_used": select_model(complexity).model_name if select_model(complexity) else "",
            "simulation_output": None,
            "status": "failed",
            "error": f"preprocess_input error: {exc}",
        }

    # Select model early so we can include its name in any error responses
    try:
        model = select_model(complexity)
        model_name = getattr(model, "model_name", str(model))
    except Exception as exc:
        logger.exception("select_model failed for complexity=%s", complexity)
        return {
            "model_used": "",
            "simulation_output": None,
            "status": "failed",
            "error": f"select_model error: {exc}",
        }

    # Attempt to enqueue the task with timeout and retries
    last_exc: Optional[BaseException] = None
    backoff_base = 0.5
    for attempt in range(0, max_retries + 1):
        try:
            # Use asyncio.wait_for to bound how long we wait for each attempt
            result = await asyncio.wait_for(
                enqueue_task(processed, complexity=complexity), timeout=timeout
            )
            return {
                "model_used": model_name,
                "simulation_output": result,
                "status": "success",
                "error": None,
            }
        except asyncio.TimeoutError as exc:
            last_exc = exc
            logger.warning(
                "enqueue_task timed out (attempt %d/%d, timeout=%ss) for model=%s",
                attempt + 1,
                max_retries + 1,
                timeout,
                model_name,
            )
            # On timeout, don't retry indefinitely — proceed to next attempt if available
        except Exception as exc:
            last_exc = exc
            logger.exception(
                "enqueue_task failed (attempt %d/%d) for model=%s",
                attempt + 1,
                max_retries + 1,
                model_name,
            )

        # If we have more attempts available, sleep with exponential backoff
        if attempt < max_retries:
            backoff = backoff_base * (2 ** attempt)
            logger.info("Retrying enqueue_task after %.2fs (attempt %d)", backoff, attempt + 2)
            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                logger.info("simulate_attack cancelled during backoff")
                return {
                    "model_used": model_name,
                    "simulation_output": None,
                    "status": "failed",
                    "error": "operation cancelled",
                }

    # If we exhausted retries, return a failure shape with details
    logger.error("enqueue_task failed after %d attempts: %s", max_retries + 1, last_exc)
    err_msg = str(last_exc) if last_exc is not None else "unknown error"
    # Distinguish timeout vs other errors if possible
    status = "timeout" if isinstance(last_exc, asyncio.TimeoutError) else "failed"
    return {
        "model_used": model_name,
        "simulation_output": None,
        "status": status,
        "error": err_msg,
    }
