# ai_core/threat_analyzer.py
import asyncio
import logging
import time
from typing import Any, Dict, Iterable

from .predictive_model import enqueue_task, select_model
from .embeddings_service import generate_embeddings
from .utils import preprocess_input

logger = logging.getLogger(__name__)

# Allowed complexity values — keep this authoritative and easy to extend
ALLOWED_COMPLEITIES = {"low", "medium", "high"}


async def analyze_threat(text: str, complexity: str = "medium", *, timeout: float = 30.0) -> Dict[str, Any]:
    """
    Analyze a piece of text for threats.

    Improvements made:
    - Input validation for `text` and `complexity`.
    - Concurrent execution: runs the predictive analysis (async) and embedding generation (potentially blocking)
      concurrently to reduce wall-clock time.
    - Uses asyncio.to_thread for embedding generation to avoid blocking the event loop if generate_embeddings is sync.
    - Timeout support to avoid hanging calls.
    - Structured return with a small `meta` section for observability (processing time, processed length).
    - Robust logging and exception handling to aid debugging in production.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    complexity = (complexity or "medium").lower()
    if complexity not in ALLOWED_COMPLEXITIES:
        logger.warning("Unsupported complexity '%s', falling back to 'medium'", complexity)
        complexity = "medium"

    processed = preprocess_input(text)

    # Resolve model synchronously (cheap) so we can include the model name in logs/response
    model = select_model(complexity)
    model_name = getattr(model, "model_name", str(model))

    # Prepare concurrent work:
    # - enqueue_task is async and awaited directly.
    # - generate_embeddings is often sync; run it in a thread to avoid blocking the event loop.
    analysis_coro = enqueue_task(processed, complexity)
    embeddings_task = asyncio.to_thread(generate_embeddings, [processed])

    start = time.perf_counter()
    try:
        analysis_result, embeddings_result = await asyncio.wait_for(
            asyncio.gather(analysis_coro, embeddings_task), timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.exception("analyze_threat timed out after %.1fs (model=%s)", timeout, model_name)
        raise
    except Exception:
        logger.exception("analyze_threat failed (model=%s)", model_name)
        raise

    # normalize embedding extraction
    embedding = None
    if isinstance(embeddings_result, (list, tuple)):
        if embeddings_result:
            embedding = embeddings_result[0]
    else:
        embedding = embeddings_result

    elapsed = time.perf_counter() - start

    logger.debug(
        "analyze_threat completed (model=%s, complexity=%s, elapsed=%.3fs, processed_len=%d)",
        model_name,
        complexity,
        elapsed,
        len(processed),
    )

    return {
        "model_used": model_name,
        "analysis": analysis_result,
        "embedding": embedding,
        "meta": {
            "processed_text_length": len(processed),
            "complexity": complexity,
            "elapsed_seconds": elapsed,
        },
    }
