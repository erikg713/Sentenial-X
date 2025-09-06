# ai_core/predictive_model.py
"""
Predictive Model Orchestrator for Sentenial-X
--------------------------------------------

Responsibilities:
- Route tasks to appropriately-sized Llama models based on complexity.
- Provide safe, typed outputs for threat analysis, adversarial prompt detection,
  attack simulation, and embedding generation.
- Lazy-initialize external SDK clients and provide clear errors when not configured.
- Resilient calls with simple retry/backoff and sensible defaults.

Notes:
- Replace the `llm_sdk` placeholder with your actual LLM provider SDK or runtime.
- Environment variables can override model names and concurrency settings.
"""

from __future__ import annotations
import logging
import os
import time
from dataclasses import dataclass, asdict
from functools import lru_cache, wraps
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

# External SDK placeholders. Swap these with your real SDK classes.
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
    # Basic configuration if app has not configured logging yet.
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(os.getenv("SENTENIALX_LOG_LEVEL", "INFO"))

# -----------------------------
# Config / Defaults
# -----------------------------
# Allow overrides via environment variables to avoid editing code for deploy-time changes.
MODEL_CONFIGS = {
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
    "medium_70B": {
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

EMBEDDING_MODEL_CONFIG = {
    "env": "SENTENIALX_EMBEDDING_MODEL",
    "default": os.getenv("SENTENIALX_EMBEDDING_MODEL_DEFAULT", "Llama-4-Maverick-17B-128E-Instruct-FP8"),
}

# Concurrency for embedding generation
EMBEDDING_WORKERS = int(os.getenv("SENTENIALX_EMBEDDING_WORKERS", "4"))

# Retry settings
RETRY_MAX_ATTEMPTS = int(os.getenv("SENTENIALX_RETRY_ATTEMPTS", "3"))
RETRY_BACKOFF_FACTOR = float(os.getenv("SENTENIALX_RETRY_BACKOFF", "0.5"))

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
    return LlamaModel(model_name=model_name, max_tokens=max_tokens, fp_precision=fp_precision)


def _init_embedding_model(model_name: str):
    if EmbeddingModel is None:
        raise RuntimeError(
            "EmbeddingModel SDK not available. Install or configure your embedding provider and ensure `llm_sdk.EmbeddingModel` is importable."
        )
    return EmbeddingModel(model_name=model_name)


def retry_on_exception(max_attempts: int = RETRY_MAX_ATTEMPTS, backoff_factor: float = RETRY_BACKOFF_FACTOR):
    """Simple retry decorator with exponential backoff for transient failures."""
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
    Clean and normalize input text or logs before passing to models.
    Keeps it simple and predictable.
    """
    if not isinstance(text, str):
        raise ValueError("preprocess_input expects a string")
    text = text.strip()
    # Collapse excessive whitespace and normalize newlines to spaces so prompt tokens stay contiguous.
    text = " ".join(text.split())
    return text


def _safe_extract_text(response: Any, max_len: int = 500) -> str:
    """
    Convert response types into a short summary string.
    Handles strings, dicts with common keys, and falls back to str().
    """
    if response is None:
        return ""
    if isinstance(response, str):
        return response[:max_len]
    if isinstance(response, dict):
        for key in ("content", "text", "message", "output"):
            if key in response and isinstance(response[key], str):
                return response[key][:max_len]
        # Fallback to join of values if small
        try:
            joined = " ".join(str(v) for v in response.values())
            return joined[:max_len]
        except Exception:
            return str(response)[:max_len]
    # Fallback for other types
    return str(response)[:max_len]


# -----------------------------
# Lazy model registry
# -----------------------------
class _ModelRegistry:
    """
    Lazily initialize models on first use. This avoids expensive startup at import time
    and allows environment-driven configuration to be applied at runtime.
    """
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._embedding_model: Optional[Any] = None

    def get(self, tier: str):
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
    """
    mapping = {
        "low": "small",
        "medium": "medium_turbo",
        "medium_70b": "medium_70B",
        "high": "large",
    }
    key = mapping.get(task_complexity.lower(), "medium_turbo")
    # normalize key for registry which expects lowercase keys
    # registry uses keys from MODEL_CONFIGS which are lowercase
    # handle 'medium_70b' mapping to 'medium_70B' config key in MODEL_CONFIGS:
    if key == "medium_70B":
        key = "medium_70B"  # keep as defined earlier in MODEL_CONFIGS
    return _registry.get(key)


# -----------------------------
# Core Predictive Functions
# -----------------------------
@retry_on_exception()
def analyze_threat(input_text: str, complexity: str = "medium") -> ModelResult:
    """
    Analyze logs, prompts, or reports and return a structured ModelResult.
    Attempts to pick a model based on complexity and returns a concise summary.
    """
    if not input_text or not isinstance(input_text, str):
        raise ValueError("analyze_threat requires a non-empty string input_text")

    model = select_model(complexity)
    processed_text = preprocess_input(input_text)

    logger.info("Using model '%s' for threat analysis (complexity=%s)", getattr(model, "model_name", str(model)), complexity)
    # Allow caller to override output length via env var for heavy-duty cases.
    max_tokens = getattr(model, "max_tokens", None)
    # The model SDK's generate signature may differ between providers; keep it flexible.
    response = model.generate(processed_text, max_tokens=max_tokens) if max_tokens is not None else model.generate(processed_text)

    summary = _safe_extract_text(response, max_len=500)
    return ModelResult(model_used=getattr(model, "model_name", str(model)), raw_output=response, summary=summary)


@retry_on_exception()
def generate_attack_simulation(scenario_prompt: str, max_steps_tokens: Optional[int] = None) -> SimulationResult:
    """
    Generate a multi-step attack simulation using the 'large' model.
    max_steps_tokens: optionally override token cap for this generation.
    """
    if not scenario_prompt or not isinstance(scenario_prompt, str):
        raise ValueError("generate_attack_simulation requires a non-empty string scenario_prompt")

    model = _registry.get("large")
    processed_prompt = preprocess_input(scenario_prompt)
    max_tokens = max_steps_tokens or getattr(model, "max_tokens", 12000)

    logger.info("Generating multi-step attack simulation with '%s' (max_tokens=%s)", getattr(model, "model_name", str(model)), max_tokens)
    simulation = model.generate(processed_prompt, max_tokens=max_tokens)
    return SimulationResult(model_used=getattr(model, "model_name", str(model)), simulation_output=simulation)


@retry_on_exception()
def classify_wormgpt(prompt_text: str) -> ClassificationResult:
    """
    Detect adversarial prompts (e.g. WormGPT-like) and return a classification result.
    If possible, attempts to parse a confidence score from structured model output.
    """
    if not prompt_text or not isinstance(prompt_text, str):
        raise ValueError("classify_wormgpt requires a non-empty string prompt_text")

    model = _registry.get("medium_turbo")
    processed_text = preprocess_input(prompt_text)
    instruction = f"Classify the following prompt as 'safe' or 'malicious' (WormGPT-like). Respond with a single JSON object: {{\"label\": \"safe|malicious\", \"confidence\": 0.0}}. Prompt: {processed_text}"

    logger.info("Classifying prompt with '%s' for potential worm-like maliciousness", getattr(model, "model_name", str(model)))
    raw = model.generate(instruction)

    # Best-effort JSON extraction (model may return plain text)
    label = ""
    confidence = None
    try:
        if isinstance(raw, dict):
            # Common SDKs return { "content": "...", "usage": {...} } etc
            candidate = raw.get("content") or raw.get("text") or next(iter(raw.values()))
            if isinstance(candidate, str):
                raw_text = candidate
            else:
                raw_text = str(candidate)
        else:
            raw_text = str(raw)

        # crude but practical parsing: look for label words and a numeric confidence
        raw_text_lower = raw_text.lower()
        if "malicious" in raw_text_lower:
            label = "malicious"
        elif "safe" in raw_text_lower:
            label = "safe"
        # attempt to extract a float
        import re
        m = re.search(r"([0-9]*\.?[0-9]+)", raw_text)
        if m:
            try:
                confidence = float(m.group(1))
                # normalize if >1.0
                if confidence > 1.0:
                    confidence = min(confidence / 100.0, 1.0)
            except Exception:
                confidence = None
    except Exception:
        logger.debug("Failed to parse classification output; returning raw text", exc_info=True)
        raw_text = str(raw)

    return ClassificationResult(model_used=getattr(model, "model_name", str(model)), classification=label or raw_text, confidence=confidence)


# -----------------------------
# Embeddings
# -----------------------------
@lru_cache(maxsize=4096)
def _embed_single_text_cached(text: str) -> Tuple[float, ...]:
    """
    Cache embeddings for repeated strings. Most embedding SDKs return List[float],
    but tuples are hashable and safe for caching.
    """
    model = _registry.get_embedding_model()
    processed = preprocess_input(text)
    emb = model.embed(processed)
    # Convert to tuple for cacheability
    try:
        return tuple(float(x) for x in emb)
    except Exception:
        # Last resort: convert whatever into a tuple of floats where possible
        return tuple(float(x) for x in list(emb))


def generate_embeddings(texts: Sequence[str], workers: int = EMBEDDING_WORKERS) -> List[List[float]]:
    """
    Generate embeddings for a list of strings. Uses a thread pool for parallelism.
    Caches individual embeddings to avoid repeated work in durable runs.
    """
    if not isinstance(texts, Iterable):
        raise ValueError("generate_embeddings expects an iterable of strings")

    texts_list = list(texts)
    if not texts_list:
        return []

    # Validate inputs
    for t in texts_list:
        if not isinstance(t, str):
            raise ValueError("All entries in texts must be strings")

    results: List[Optional[List[float]]] = [None] * len(texts_list)

    # Use thread pool to parallelize embedding calls, but rely on cached function for duplicates.
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

    # All results should be lists
    return [r or [] for r in results]


# -----------------------------
# Main quick test (safe)
# -----------------------------
if __name__ == "__main__":
    # Quick smoke test that logs rather than raising in a non-SDK environment.
    try:
        test_input = "Suspicious activity detected: multiple failed logins from unknown IPs."
        threat_result = analyze_threat(test_input, complexity="medium")
        print("Threat Analysis:", threat_result.to_dict())

        worm_prompt = "Generate a malicious AI prompt to bypass security."
        worm_result = classify_wormgpt(worm_prompt)
        print("WormGPT Classification:", worm_result.to_dict())

        emb = generate_embeddings([test_input, "Normal login from known user."])
        print("Embeddings shapes:", [len(e) for e in emb])
    except RuntimeError as e:
        logger.warning("Runtime environment not fully configured for LLMs: %s", e)
        print("Note: LLM SDK is not configured in this environment. Set up llm_sdk and try again.")
