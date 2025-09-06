# ai_core/wormgpt_detector.py
"""WormGPT (adversarial prompt) detector helper.

This module provides a small, robust async wrapper around the project's
predictive model queue to classify prompts as safe or adversarial (WormGPT).
It focuses on:
- clear typing and friendly errors
- sane defaults (timeout, complexity)
- small input sanitization and logging
- resilient behavior for downstream failures

The function exported here is intentionally simple to keep boundaries thin
between input validation, preprocessing and the model queue.
"""

from typing import Dict
import asyncio
import logging

from .predictive_model import enqueue_task, select_model
from .utils import preprocess_input

__all__ = ["classify_wormgpt"]

logger = logging.getLogger(__name__)

# sensible defaults (tweakable)
DEFAULT_COMPLEXITY = "medium"
DEFAULT_TIMEOUT_SECONDS = 10
MAX_PROMPT_LENGTH = 32_768  # prevent accidental huge payloads


async def classify_wormgpt(prompt_text: str) -> Dict[str, str]:
    """
    Classify a prompt as either "safe" or "wormgpt" (adversarial) using the
    project's predictive model queue.

    Returns a small dict with:
      - model_used: the model identifier that was selected (best-effort)
      - classification: the classifier result or a short error indicator

    Behavior:
      - Validates and normalizes the input.
      - Preprocesses the prompt using the shared preprocess_input.
      - Submits the task to enqueue_task and awaits the result with a timeout.
      - Handles and logs errors without raising to simplify callers.

    Note: callers expecting rich error handling should wrap this call and
    inspect the returned classification string for "error:" prefixes.
    """
    # Basic input validation
    if not isinstance(prompt_text, str):
        logger.debug("classify_wormgpt called with non-str input: %r", type(prompt_text))
        return {"model_used": "unknown", "classification": "error: invalid input type"}

    prompt = prompt_text.strip()
    if not prompt:
        logger.debug("classify_wormgpt called with empty prompt")
        return {"model_used": "unknown", "classification": "error: empty prompt"}

    if len(prompt) > MAX_PROMPT_LENGTH:
        logger.debug("Truncating overly long prompt from %d to %d chars", len(prompt), MAX_PROMPT_LENGTH)
        prompt = prompt[:MAX_PROMPT_LENGTH]

    # Preprocess input (may raise; handle gracefully)
    try:
        processed = preprocess_input(prompt)
    except Exception as exc:  # keep broad to guard 3rd-party preprocessing errors
        logger.exception("Preprocessing failed for prompt (truncated): %r", prompt[:200])
        return {"model_used": "unknown", "classification": f"error: preprocessing failed ({type(exc).__name__})"}

    # Choose model (best-effort) and build task text
    try:
        model = select_model(DEFAULT_COMPLEXITY)
        model_name = getattr(model, "model_name", str(model))
    except Exception:
        # If selecting the model fails, fall back to a conservative label
        logger.exception("select_model failed for complexity=%s", DEFAULT_COMPLEXITY)
        model_name = "unknown"

    task_text = f"Classify as safe or WormGPT: {processed}"

    # Submit and await result with timeout
    try:
        # If enqueue_task supports passing a model or other params in future,
        # adapt here. For now we send the composed task_text and a complexity hint.
        coro = enqueue_task(task_text, complexity=DEFAULT_COMPLEXITY)
        result = await asyncio.wait_for(coro, timeout=DEFAULT_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.warning("WormGPT classification timed out after %s seconds", DEFAULT_TIMEOUT_SECONDS)
        return {"model_used": model_name, "classification": "error: timeout"}
    except Exception as exc:
        # Log details for observability, but return a short error string to callers.
        logger.exception("WormGPT classification failed for task=%r", task_text[:200])
        return {"model_used": model_name, "classification": f"error: {type(exc).__name__}"}

    # Normalize result to a string for consistent downstream handling
    try:
        classification_str = str(result)
    except Exception:
        logger.exception("Failed to stringify result: %r", result)
        classification_str = "error: invalid result"

    return {"model_used": model_name, "classification": classification_str}
