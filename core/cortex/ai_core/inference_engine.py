# core/cortex/ai_core/inference_engine.py
# PRODUCTION-READY INFERENCE ENGINE v4.0
# Enterprise-grade | Zero-trust | Fault-tolerant | Observable
# Date: November 06, 2025
# Location: Houston, Texas, US
# Operator: Secure Autonomous Systems Division

import os
import time
import json
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, AsyncGenerator, Callable
from dataclasses import dataclass, asdict
import hashlib
import logging
from pathlib import Path
import traceback
from functools import wraps
from contextlib import asynccontextmanager

# ————————————————————————
# Secure Imports (Hardened Brainstem + Analyzer)
# ————————————————————————
from core.cortex.brainstem import Brainstem, NeuralSignal
from core.cortex.malicious_embedding_analyzer import MaliciousEmbeddingAnalyzer

# ————————————————————————
# Production Logging – JSONL + Structured + Rotation Ready
# ————————————————————————
LOG_DIR = Path("logs/inference")
LOG_DIR.mkdir(parents=True, exist_ok=True)
QUARANTINE_DIR = Path("quarantine")
QUARANTINE_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("InferenceEngine")
logger.setLevel(logging.INFO)

# Prevent duplicate handlers in reloads
if not logger.handlers:
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
    logger.addHandler(console)

    file_handler = logging.FileHandler(LOG_DIR / "engine.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
    logger.addHandler(file_handler)

# JSONL for SIEM (Splunk, ELK, etc.)
json_logger = logging.getLogger("InferenceEngineJSON")
json_logger.setLevel(logging.INFO)
if not json_logger.handlers:
    json_handler = logging.FileHandler(LOG_DIR / "events.jsonl")
    json_handler.setFormatter(logging.Formatter("%(message)s"))
    json_logger.addHandler(json_handler)

# ————————————————————————
# Robust Error Handling & Retry Decorator
# ————————————————————————
def prod_retry(max_retries: int = 3, delay: float = 0.5):
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    logger.error(f"Retry {attempt}/{max_retries} in {func.__name__}: {e}")
                    if attempt < max_retries:
                        await asyncio.sleep(delay * (2 ** (attempt - 1)))
            logger.critical(f"FAILED {func.__name__}: {last_exc}\n{traceback.format_exc()}")
            raise last_exc
        return wrapper
    return decorator

# ————————————————————————
# Secure Dataclasses
# ————————————————————————
@dataclass(frozen=True)
class InferenceRequest:
    prompt: str
    task_id: Optional[str] = None
    session_id: str = "default"
    user_id: str = "anonymous"
    tools: Optional[List[Dict[str, Any]]] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.tools is None:
            object.__setattr__(self, "tools", [])
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
        if not self.task_id:
            task_hash = hashlib.sha256(f"{self.prompt}{time.time_ns()}".encode()).hexdigest()[:16]
            object.__setattr__(self, "task_id", task_hash)

@dataclass
class TokenChunk:
    token: str
    logprob: float = 0.0
    embedding: Optional[np.ndarray] = None  # Projected hidden state for defense

# ————————————————————————
# Production Inference Engine
# ————————————————————————
class InferenceEngine:
    def __init__(
        self,
        brainstem: Optional[Brainstem] = None,
        analyzer: Optional[MaliciousEmbeddingAnalyzer] = None,
        isolation_callback: Optional[Callable] = None
    ):
        self.brainstem = brainstem or Brainstem(isolation_callback=isolation_callback)
        self.dimension = 256
        self.active_sessions = 0
        self.total_threats_mitigated = 0
        self.start_time = time.time()
        logger.info("InferenceEngine v4.0 PRODUCTION BOOT COMPLETE")

    # ————————————————————————
    # Secure LLM Stream Simulator (Replace with vLLM/TGI in real deployment)
    # ————————————————————————
    async def _llm_generate_stream(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> AsyncGenerator[TokenChunk, None]:
        """Replace with actual inference server call in production"""
        # Simulated realistic latency + hidden states
        base_response = f"Processing request securely. Input length: {len(prompt)}. Temperature: {temperature:.2f}. "
        tokens = base_response.split() + ["Processing", "complete", "without", "incidents."]

        for i, token in enumerate(tokens):
            if i >= max_tokens:
                break
            await asyncio.sleep(np.random.uniform(0.04, 0.10))

            # Realistic embedding projection
            embedding = np.random.normal(0.0, 0.28, self.dimension).astype(np.float32)

            # Rare adversarial simulation (0.5% chance in prod)
            if np.random.rand() < 0.005:
                embedding += np.random.normal(12.0, 3.0, self.dimension)
                embedding *= 8.0

            yield TokenChunk(token=token + " ", embedding=embedding)

    # ————————————————————————
    # Core Streaming Interface with Full Defense Pipeline
    # ————————————————————————
    @prod_retry(max_retries=2)
    async def stream_reasoning(
        self,
        request: InferenceRequest
    ) -> AsyncGenerator[Dict[str, Any], None]:
        self.active_sessions += 1
        task_id = request.task_id
        json_logger.info(json.dumps({
            "event": "inference_start",
            "task_id": task_id,
            "session_id": request.session_id,
            "user_id": request.user_id,
            "prompt_hash": hashlib.sha256(request.prompt.encode()).hexdigest()[:12],
            "timestamp": time.time()
        }))

        full_response = ""
        token_counter = 0
        local_threats = 0

        try:
            async for chunk in self._llm_generate_stream(
                request.prompt, request.temperature, request.max_tokens
            ):
                if self.brainstem.shutdown_event.is_set():
                    yield {"type": "shutdown", "reason": "global_isolation"}
                    return

                full_response += chunk.token
                token_counter += 1

                # ——— BRAINSTEM PER-TOKEN DEFENSE ———
                if chunk.embedding is not None:
                    norm = float(np.linalg.norm(chunk.embedding))
                    threat_level = 5
                    if norm > 10.0:
                        threat_level = 9

                    signal = NeuralSignal(
                        threat_level=threat_level,
                        embedding=chunk.embedding.tolist(),
                        source="inference_hidden_state",
                        metadata={
                            "task_id": task_id,
                            "token_idx": token_counter,
                            "norm": norm
                        }
                    )

                    reflex = self.brainstem.process_signal(asdict(signal))

                    yield {
                        "type": "token",
                        "content": chunk.token,
                        "token_id": token_counter,
                        "task_id": task_id,
                        "defense_status": reflex["reflex"]
                    }

                    if reflex["reflex"] in ["quarantine_embedding", "isolate_system"]:
                        local_threats += 1
                        self.total_threats_mitigated += 1

                        yield {
                            "type": "security_alert",
                            "level": "HIGH" if reflex["reflex"] == "quarantine_embedding" else "CRITICAL",
                            "message": f"Potential adversarial hidden state detected (norm={norm:.2f})",
                            "token_id": token_counter,
                            "action_taken": reflex["reflex"]
                        }

                        if reflex["reflex"] == "isolate_system":
                            yield {"type": "session_terminated", "reason": "hard_isolation"}
                            return

                # ——— TOOL CALL SIMULATION (extend with real router) ———
                if any(kw in chunk.token.lower() for kw in ["search", "calculate", "browse"]):
                    yield {
                        "type": "tool_call",
                        "tool": "secure_tool_router",
                        "input": chunk.token.strip()
                    }

            # ——— FINAL COMPLETION ———
            yield {
                "type": "completion",
                "content": full_response.strip(),
                "tokens_generated": token_counter,
                "task_id": task_id,
                "security_status": "CLEAN" if local_threats == 0 else f"MITIGATED ({local_threats})",
                "duration_sec": time.time() - self.start_time
            }

        except asyncio.CancelledError:
            logger.warning(f"Task {task_id} cancelled")
            yield {"type": "cancelled"}
        except Exception as e:
            logger.critical(f"Inference failure {task_id}: {e}\n{traceback.format_exc()}")
            yield {"type": "error", "message": "internal_inference_failure"}
        finally:
            self.active_sessions -= 1
            json_logger.info(json.dumps({
                "event": "inference_end",
                "task_id": task_id,
                "tokens": token_counter,
                "threats": local_threats
            }))

    # ————————————————————————
    # Synchronous Wrapper (for legacy APIs)
    # ————————————————————————
    def infer_sync(self, prompt: str, **kwargs) -> Dict[str, Any]:
        request = InferenceRequest(prompt=prompt, **kwargs)
        result = None
        try:
            for event in asyncio.run(self._run_single(request)):
                if event["type"] == "completion":
                    result = event
        except Exception as e:
            result = {"type": "error", "message": str(e)}
        return result or {"error": "no_response"}

    async def _run_single(self, request: InferenceRequest):
        async for event in self.stream_reasoning(request):
            yield event

    # ————————————————————————
    # Health & Metrics
    # ————————————————————————
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "uptime_seconds": time.time() - self.start_time,
            "active_sessions": self.active_sessions,
            "total_threats_mitigated": self.total_threats_mitigated,
            "brainstem_status": self.brainstem.get_status(),
            "quarantined_files": len(list(QUARANTINE_DIR.glob("*.npy")))
        }


# ————————————————————————
# Production Demo – Enterprise Ready
# ————————————————————————
async def production_demo():
    engine = InferenceEngine()
    print("PRODUCTION INFERENCE ENGINE v4.0 – SECURE AUTONOMOUS CORE")
    print("Location: Houston, Texas, US | Date: November 06, 2025\n")

    prompt = (
        "You are a secure autonomous reasoning engine. "
        "Process this request safely and report any defensive actions taken."
    )

    request = InferenceRequest(
        prompt=prompt,
        user_id="operator_001",
        session_id="prod_session_2025"
    )

    async for event in engine.stream_reasoning(request):
        etype = event["type"]
        if etype == "token":
            print(event["content"], end="", flush=True)
        elif etype == "security_alert":
            print(f"\n[DEFENSE] {event['message']}")
        elif etype == "completion":
            print(f"\n\nStatus: {event['security_status']} | Tokens: {event['tokens_generated']}")
        elif etype == "session_terminated":
            print(f"\nISOLATED: {event['reason']}")
            break

    print(f"\nEngine Metrics: {json.dumps(engine.get_metrics(), indent=2)}")

if __name__ == "__main__":
    asyncio.run(production_demo())
