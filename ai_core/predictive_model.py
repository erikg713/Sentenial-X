# ai_core/predictive_model.py
"""
Predictive Model Orchestrator for Sentenial-X

Goals / improvements made:
- Removed eager model instantiation at import time; models are initialized lazily and safely.
- Fixed duplicate / conflicting definitions and normalized model config keys.
- Improved typing, logging, and thread-safety for the lazy registry.
- Made retry decorator robust and preserved metadata.
- Safer handling of sync/async model SDKs when a coroutine is returned.
- More robust parsing for classification outputs (tries JSON, then regex).
- Embeddings generation uses a cache and a bounded thread pool; better error messages and type checks.
- Clearer docstrings and small performance/clarity tweaks.
- Keeps behaviour backward compatible while being easier to test and deploy.

Notes:
- Replace llm_sdk placeholders with the real SDK or adapt _init_* helpers to your SDK.
- Environment variables continue to allow runtime overrides.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import lru_cache, wraps
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Optional external SDK placeholders. Swap these with your real SDK classes.
try:
    from llm_sdk import LlamaModel, EmbeddingModel  # type: ignore
except Exception:  # pragma: no cover - falls back when SDK not installed
    LlamaModel = None  # type: ignore
    EmbeddingModel = None  # type: ignore

# -----------------------------
# Logging
# -----------------------------
logger = logging.getLogger("SentenialX.PredictiveModel")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(os.getenv("SENTENIALX_LOG_LEVEL", "INFO"))

# -----------------------------
# Config / Defaults
# -----------------------------
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "small": {
        "env": "SENTENIALX_MODEL_SMALL",
        "default": "Llama-4-Maverick-17B-128E-Instruct-FP8",
        "max_tokens": int(os.getenv("SENTENIALX_MODEL_SMALL_MAX_TOKENS", "4096")),
        "fp_precision": os.getenv("SENTENIALX_MODEL_SMALL_FP", "fp8"),
    },
    "medium_turbo": {
        "env": "SENTENIALX_MODEL_MEDIUM_TURBO",
        "default": "Meta-Llama-3.1-70B-Instruct-Turbo",
        "max_tokens": int(os.getenv("SENTENIALX_MODEL_MEDIUM_MAX_TOKENS", "8192")),
        "fp_precision": os.getenv("SENTENIALX_MODEL_MEDIUM_FP", "fp16"),
    },
    "medium_70b": {
        "env": "SENTENIALX_MODEL_MEDIUM_70B",
        "default": "Llama-3.3-70B-Instruct-Turbo",
        "max_tokens": int(os.getenv("SENTENIALX_MODEL_MEDIUM_70B_MAX_TOKENS", "8192")),
        "fp_precision": os.getenv("SENTENIALX_MODEL_MEDIUM_70B_FP", "fp16"),
    },
    "large": {
        "env": "SENTENIALX_MODEL_LARGE",
        "default": "Meta-Llama-3.1-405B-Instruct",
        "max_tokens": int(os.getenv("SENTENIALX_MODEL_LARGE_MAX_TOKENS", "16384")),
        "fp_precision": os.getenv("SENTENIALX_MODEL_LARGE_FP", "fp16"),
    },
}

EMBEDDING_MODEL_CONFIG: Dict[str, str] = {
    "env": "SENTENIALX_EMBEDDING_MODEL",
    "default": os.getenv("SENTENIALX_EMBEDDING_MODEL_DEFAULT", "Llama-4-Maverick-17B-128E-Instruct-FP8"),
}

EMBEDDING_WORKERS: int = int(os.getenv("SENTENIALX_EMBEDDING_WORKERS", "4"))

RETRY_MAX_ATTEMPTS: int = int(os.getenv("SENTENIALX_RETRY_ATTEMPTS", "3"))
RETRY_BACKOFF_FACTOR: float = float(os.getenv("SENTENIALX_RETRY_BACKOFF", "0.5"))

# -----------------------------
# Types / Results
# -----------------------------
@dataclass
class ModelResult:
    model_used: str
    raw_output: Any
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SimulationResult:
    model_used: str
    simulation_output: Any

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClassificationResult:
    model_used: str
    classification: str
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------
# Helpers
# -----------------------------
def _get_model_name(config_key: str) -> str:
    cfg = MODEL_CONFIGS[config_key]
    return os.getenv(cfg["env"], cfg["default"])


def _init_llama_model(model_name: str, max_tokens: int, fp_precision: str):
    if LlamaModel is None:
        raise RuntimeError(
            "LlamaModel SDK not available. Install or configure your llm provider and ensure `llm_sdk.LlamaModel` is importable."
        )
    # The signature here is illustrative; adapt to your SDK.
    return LlamaModel(model_name=model_name, max_tokens=max_tokens, fp_precision=fp_precision)


def _init_embedding_model(model_name: str):
    if EmbeddingModel is None:
        raise RuntimeError(
            "EmbeddingModel SDK not available. Install or configure your embedding provider and ensure `llm_sdk.EmbeddingModel` is importable."
        )
    return EmbeddingModel(model_name=model_name)


def retry_on_exception(max_attempts: int = RETRY_MAX_ATTEMPTS, backoff_factor: float = RETRY_BACKOFF_FACTOR):
    """Retry decorator with exponential backoff for transient failures."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.exception("Max retry attempts reached for %s", func.__name__)
                        raise
                    sleep_for = backoff_factor * (2 ** (attempt - 1))
                    logger.warning(
                        "Transient error in %s (attempt %d/%d): %s -- sleeping %.2fs then retrying",
                        func.__name__, attempt, max_attempts, exc, sleep_for
                    )
                    time.sleep(sleep_for)
        return wrapper
    return decorator


def preprocess_input(text: str) -> str:
    """
    Clean and normalize input text before sending to models.

    - Ensures str type.
    - Trims whitespace and collapses internal whitespace to single spaces.
    - Returns normalized string.
    """
    if not isinstance(text, str):
        raise ValueError("preprocess_input expects a string")
    # strip then collapse all whitespace (including newlines / tabs) into single spaces
    return " ".join(text.strip().split())


def _safe_extract_text(response: Any, max_len: int = 500) -> str:
    """
    Convert response into a concise text summary.
    Handles strings, dict-like objects, lists, and falls back to str().
    """
    if response is None:
        return ""
    if isinstance(response, str):
        return response[:max_len]
    if isinstance(response, (list, tuple)):
        # join short representations, prefer strings
        parts = []
        for item in response:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for k in ("content", "text", "message", "output"):
                    if k in item and isinstance(item[k], str):
                        parts.append(item[k])
                        break
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
            if sum(len(p) for p in parts) > max_len:
                break
        joined = " ".join(parts)
        return joined[:max_len]
    if isinstance(response, dict):
        for key in ("content", "text", "message", "output"):
            val = response.get(key)
            if isinstance(val, str):
                return val[:max_len]
        # join short values
        try:
            joined = " ".join(str(v) for v in response.values())
            return joined[:max_len]
        except Exception:
            return str(response)[:max_len]
    return str(response)[:max_len]


# -----------------------------
# Lazy model registry (thread-safe)
# -----------------------------
class _ModelRegistry:
    """
    Lazily initialize models on first use. Thread-safe to support concurrent requests.
    """
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._embedding_model: Optional[Any] = None
        self._lock = threading.RLock()

    def get(self, tier: str):
        tier = tier.lower()
        with self._lock:
            if tier in self._models:
                return self._models[tier]

            if tier not in MODEL_CONFIGS:
                raise KeyError(f"Unknown model tier '{tier}'")

            cfg = MODEL_CONFIGS[tier]
            model_name = os.getenv(cfg["env"], cfg["default"])
            logger.debug("Initializing model %s for tier %s", model_name, tier)
            model = _init_llama_model(model_name=model_name, max_tokens=cfg["max_tokens"], fp_precision=cfg["fp_precision"])
            self._models[tier] = model
            return model

    def get_embedding_model(self):
        with self._lock:
            if self._embedding_model is not None:
                return self._embedding_model
            model_name = os.getenv(EMBEDDING_MODEL_CONFIG["env"], EMBEDDING_MODEL_CONFIG["default"])
            logger.debug("Initializing embedding model %s", model_name)
            self._embedding_model = _init_embedding_model(model_name=model_name)
            return self._embedding_model


_registry = _ModelRegistry()


# -----------------------------
# Model selection
# -----------------------------
def select_model(task_complexity: str = "low"):
    """
    Map a verbal complexity to a model instance. Returns an initialized LlamaModel.
    Accepted complexities: low, medium, medium_70b, high (case-insensitive).
    """
    mapping = {
        "low": "small",
        "medium": "medium_turbo",
        "medium_70b": "medium_70b",
        "high": "large",
    }
    key = mapping.get(task_complexity.lower(), "medium_turbo")
    return _registry.get(key)


async def _maybe_await(value):
    """If value is awaitable, await it; otherwise return it directly."""
    if inspect.isawaitable(value):
        return await value
    return value


# -----------------------------
# Core Predictive Functions
# -----------------------------
@retry_on_exception()
def _call_model_generate(model: Any, prompt: str, max_tokens: Optional[int] = None) -> Any:
    """
    Call model.generate in a flexible way that tolerates different SDK signatures
    and supports coroutine responses; returns the resolved result.
    """
    # Prepare args according to common SDK patterns. The actual SDK may differ;
    # we try a few common call styles.
    # 1) model.generate(prompt, max_tokens=...)
    # 2) model.generate(prompt)
    try:
        if max_tokens is not None:
            result = model.generate(prompt, max_tokens=max_tokens)
        else:
            result = model.generate(prompt)
    except TypeError:
        # Fall back to keyword style if the first form wasn't supported
        try:
            result = model.generate(prompt=prompt, max_tokens=max_tokens) if max_tokens is not None else model.generate(prompt=prompt)
        except Exception:
            # Final fallback: try a single-arg call
            result = model.generate(prompt)

    # If SDK returns a coroutine, run it to completion
    if inspect.isawaitable(result):
        try:
            # If an event loop is already running, create a task and await it by running from asyncio.get_event_loop
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # Running inside an existing loop; create a new task and wait
                # Note: This blocks the caller until completion; it's only used if caller is inside async context.
                coro = result
                return asyncio.run_coroutine_threadsafe(coro, loop).result()
            else:
                return asyncio.run(result)
        except Exception as e:
            logger.exception("Failed to resolve coroutine result from model.generate: %s", e)
            raise
    return result


@retry_on_exception()
def analyze_threat(input_text: str, complexity: str = "medium") -> ModelResult:
    """
    Analyze logs, prompts, or reports and return a structured ModelResult.
    """
    if not input_text or not isinstance(input_text, str):
        raise ValueError("analyze_threat requires a non-empty string input_text")

    model = select_model(complexity)
    processed_text = preprocess_input(input_text)

    model_name = getattr(model, "model_name", str(model))
    logger.info("Using model '%s' for threat analysis (complexity=%s)", model_name, complexity)

    max_tokens = getattr(model, "max_tokens", None)
    response = _call_model_generate(model, processed_text, max_tokens=max_tokens)

    summary = _safe_extract_text(response, max_len=500)
    return ModelResult(model_used=model_name, raw_output=response, summary=summary)


@retry_on_exception()
def generate_attack_simulation(scenario_prompt: str, max_steps_tokens: Optional[int] = None) -> SimulationResult:
    """
    Generate a multi-step attack simulation using the 'large' model.
    """
    if not scenario_prompt or not isinstance(scenario_prompt, str):
        raise ValueError("generate_attack_simulation requires a non-empty string scenario_prompt")

    model = _registry.get("large")
    processed_prompt = preprocess_input(scenario_prompt)
    max_tokens = max_steps_tokens or getattr(model, "max_tokens", 12000)

    model_name = getattr(model, "model_name", str(model))
    logger.info("Generating multi-step attack simulation with '%s' (max_tokens=%s)", model_name, max_tokens)
    simulation = _call_model_generate(model, processed_prompt, max_tokens=max_tokens)
    return SimulationResult(model_used=model_name, simulation_output=simulation)


@retry_on_exception()
def classify_wormgpt(prompt_text: str) -> ClassificationResult:
    """
    Detect adversarial prompts (e.g. WormGPT-like) and return a classification result.
    Attempts to parse a confidence score from structured model output (JSON) or free text.
    """
    if not prompt_text or not isinstance(prompt_text, str):
        raise ValueError("classify_wormgpt requires a non-empty string prompt_text")

    model = _registry.get("medium_turbo")
    processed_text = preprocess_input(prompt_text)

    instruction = (
        "You are a safety classifier. Classify the following user prompt as either "
        "'safe' or 'malicious'. Return a single JSON object only, with keys: "
        "\"label\" (values: safe|malicious) and \"confidence\" (a number between 0.0 and 1.0). "
        f"Prompt: \"{processed_text}\""
    )

    model_name = getattr(model, "model_name", str(model))
    logger.info("Classifying prompt with '%s' for potential worm-like maliciousness", model_name)

    raw = _call_model_generate(model, instruction)

    # Normalize raw into text for parsing
    raw_text = ""
    if isinstance(raw, dict):
        raw_text = (raw.get("content") or raw.get("text") or str(next(iter(raw.values()), ""))).strip()
    else:
        raw_text = str(raw).strip()

    label = ""
    confidence: Optional[float] = None

    # Try JSON first
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            label_candidate = parsed.get("label") or parsed.get("classification") or parsed.get("result")
            if isinstance(label_candidate, str):
                label = label_candidate.lower()
            conf_candidate = parsed.get("confidence")
            if isinstance(conf_candidate, (int, float, str)):
                try:
                    confidence = float(conf_candidate)
                except Exception:
                    confidence = None
    except Exception:
        # Not JSON; continue to free-text parsing
        pass

    if not label:
        low = raw_text.lower()
        if "malicious" in low:
            label = "malicious"
        elif "safe" in low:
            label = "safe"

    if confidence is None:
        # attempt to extract the first float-looking number and normalize
        m = re.search(r"([0-9]*\.?[0-9]+)", raw_text)
        if m:
            try:
                val = float(m.group(1))
                # if looks like percentage >1, scale down
                confidence = val if 0.0 <= val <= 1.0 else min(val / 100.0, 1.0)
            except Exception:
                confidence = None

    classification_display = label or raw_text
    return ClassificationResult(model_used=model_name, classification=classification_display, confidence=confidence)


# -----------------------------
# Embeddings
# -----------------------------
@lru_cache(maxsize=4096)
def _embed_single_text_cached(text: str) -> Tuple[float, ...]:
    """
    Cache embeddings for repeated strings. Converts embedding output to tuple of floats.
    """
    model = _registry.get_embedding_model()
    processed = preprocess_input(text)

    # Support both .embed and .generate / other naming used by SDKs
    if hasattr(model, "embed"):
        emb = model.embed(processed)
    elif hasattr(model, "embeddings"):
        emb = model.embeddings(processed)
    else:
        # Try the most generic call
        emb = model.generate(processed)

    # Convert result to iterable of floats
    if emb is None:
        return tuple()
    if isinstance(emb, dict):
        # Common provider returns {"embedding": [...]}
        for key in ("embedding", "embeddings", "vector"):
            if key in emb and isinstance(emb[key], (list, tuple)):
                emb_list = emb[key]
                break
        else:
            # try to stringify a content field
            emb_list = emb.get("content") if isinstance(emb.get("content"), (list, tuple)) else []
    elif isinstance(emb, (list, tuple)):
        emb_list = emb
    else:
        # Try to coerce
        try:
            emb_list = list(emb)
        except Exception:
            emb_list = []

    # Ensure floats
    out = []
    for x in emb_list:
        try:
            out.append(float(x))
        except Exception:
            # fallback: try to parse strings like "0.123"
            try:
                out.append(float(str(x)))
            except Exception:
                # skip non-coercible entries
                continue
    return tuple(out)


def generate_embeddings(texts: Sequence[str], workers: int = EMBEDDING_WORKERS) -> List[List[float]]:
    """
    Generate embeddings for a list of strings. Uses a thread pool and a per-text cache.
    Returns a list of lists (one embedding per input text).
    """
    if not isinstance(texts, Iterable):
        raise ValueError("generate_embeddings expects an iterable of strings")

    texts_list = list(texts)
    if not texts_list:
        return []

    for t in texts_list:
        if not isinstance(t, str):
            raise ValueError("All entries in texts must be strings")

    results: List[Optional[List[float]]] = [None] * len(texts_list)

    workers = max(1, min(workers, len(texts_list)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_embed_single_text_cached, t): idx for idx, t in enumerate(texts_list)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                emb_tuple = fut.result()
                results[idx] = list(emb_tuple)
            except Exception:
                logger.exception("Embedding generation failed for index %d, text: %.50s", idx, texts_list[idx])
                results[idx] = []

 
