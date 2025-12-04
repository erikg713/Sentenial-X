core/cortex/decision_engine.py
import json
import logging
import os
from collections import deque
from dataclasses import asdict, dataclass, field
from threading import Lock
from typing import Callable, Deque, Dict, Iterable, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class Decision:
    action: str = "none"
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Union[str, float]]:
        return asdict(self)


@dataclass
class ThreatRecord:
    signal: Dict
    semantic: Dict
    decision: Decision


Rule = Callable[[Dict, Dict], Optional[Decision]]


class DecisionEngine:
    """
    A compact, production-ready decision engine with:
    - type hints and small dataclasses for clear state representation
    - configurable thresholds via constructor arguments or environment variables
    - a capped, thread-safe threat memory (deque) with export/import helpers
    - a simple rule pipeline allowing custom rules to be registered
    - deterministic confidence fusion between semantic and signal inputs
    """

    DEFAULT_MEMORY_SIZE = int(os.getenv("SENTENIAL_DECISION_MEMORY", "256"))
    DEFAULT_THREAT_LEVEL_SCALE = float(os.getenv("SENTENIAL_THREAT_SCALE", "10.0"))
    DEFAULT_BREACH_INTENT_CONFIDENCE = float(os.getenv("SENTENIAL_BREACH_INTENT_CONF", "0.9"))
    DEFAULT_SIGNAL_THRESHOLD = float(os.getenv("SENTENIAL_SIGNAL_THRESHOLD", "8.0"))

    def __init__(
        self,
        memory_size: int = DEFAULT_MEMORY_SIZE,
        threat_level_scale: float = DEFAULT_THREAT_LEVEL_SCALE,
        breach_intent_confidence: float = DEFAULT_BREACH_INTENT_CONFIDENCE,
        signal_threshold: float = DEFAULT_SIGNAL_THRESHOLD,
        rules: Optional[Iterable[Rule]] = None,
    ) -> None:
        self.threat_memory: Deque[ThreatRecord] = deque(maxlen=memory_size)
        self._lock = Lock()
        self.threat_level_scale = max(1.0, float(threat_level_scale))
        self.breach_intent_confidence = float(breach_intent_confidence)
        self.signal_threshold = float(signal_threshold)
        self._rules: List[Rule] = list(rules or [])

        logger.debug(
            "DecisionEngine initialized: memory_size=%s threat_level_scale=%s breach_intent_conf=%s signal_threshold=%s",
            memory_size,
            self.threat_level_scale,
            self.breach_intent_confidence,
            self.signal_threshold,
        )

    # Public API ------------------------------------------------------------

    def register_rule(self, rule: Rule) -> None:
        """
        Register a custom rule. Rules are executed in FIFO order and can
        return a Decision to short-circuit the default decisioning logic.
        """
        if not callable(rule):
            raise TypeError("rule must be callable")
        self._rules.append(rule)
        logger.debug("Registered custom decision rule: %s", getattr(rule, "__name__", repr(rule)))

    def evaluate(self, signal: Dict, semantic_output: Dict, context: Optional[Dict] = None) -> Dict:
        """
        Evaluate a single input (signal + semantic_output) and produce a decision dict.

        The function is defensive about malformed inputs and guarantees a
        stable, serializable dict response with "action", "confidence" and "reason".
        """
        signal = signal or {}
        semantic_output = semantic_output or {}
        context = context or {}

        # Normalize inputs (defensive)
        threat_level = self._safe_numeric(signal.get("threat_level", 0.0))
        intent = str(semantic_output.get("intent", "")).lower()
        semantic_conf = self._safe_numeric(semantic_output.get("confidence", 0.0))

        logger.debug("Evaluating: intent=%s semantic_conf=%s threat_level=%s context=%s", intent, semantic_conf, threat_level, context)

        # First allow custom rules to override/decide
        for rule in self._rules:
            try:
                custom_decision = rule(signal, semantic_output)
            except Exception as exc:  # keep robust: a buggy rule shouldn't crash the engine
                logger.exception("Custom rule %s raised exception: %s", getattr(rule, "__name__", repr(rule)), exc)
                custom_decision = None

            if isinstance(custom_decision, Decision):
                decision = custom_decision.to_dict()
                self._record(signal, semantic_output, custom_decision)
                logger.debug("Decision decided by custom rule: %s", decision)
                return decision
            elif custom_decision is not None:
                # allow rules to return plain dicts
                try:
                    normalized = Decision(**custom_decision)  # type: ignore[arg-type]
                    self._record(signal, semantic_output, normalized)
                    logger.debug("Decision decided by custom rule(dict): %s", normalized)
                    return normalized.to_dict()
                except Exception:
                    logger.exception("Custom rule returned invalid decision value: %s", custom_decision)

        # Default decisioning logic
        decision = Decision()
        # Calculate a fused confidence: weighted blend of semantic confidence and normalized threat level
        signal_score = min(max(threat_level / self.threat_level_scale, 0.0), 1.0)
        fused_conf = self._calibrate_confidence(semantic_conf, signal_score)

        # Intent-based high-confidence rule (breach)
        if intent == "breach" and semantic_conf >= self.breach_intent_confidence:
            decision.action = "activate_firewall_rules"
            decision.confidence = max(fused_conf, 0.0)
            decision.reason = "Detected breach intent via semantic analysis"

        # High raw signal rule (fallback)
        elif threat_level >= self.signal_threshold:
            decision.action = "terminate_process"
            decision.confidence = max(fused_conf * 0.95, 0.0)
            decision.reason = "High threat level based on raw signal"

        # Medium threat, escalate to monitoring action
        elif threat_level >= (self.signal_threshold * 0.6) or fused_conf >= 0.5:
            decision.action = "escalate_for_review"
            decision.confidence = fused_conf
            decision.reason = "Elevated risk — escalate for human review"

        else:
            decision.action = "none"
            decision.confidence = fused_conf
            decision.reason = "No actionable threat detected"

        self._record(signal, semantic_output, decision)
        logger.debug("Decision result: %s", decision)
        return decision.to_dict()

    # Helpers ---------------------------------------------------------------

    def _record(self, signal: Dict, semantic: Dict, decision: Decision) -> None:
        """Thread-safe append to the internal threat memory."""
        record = ThreatRecord(signal=signal, semantic=semantic, decision=decision)
        with self._lock:
            self.threat_memory.append(record)
        logger.debug("Recorded threat: action=%s confidence=%.3f memory_size=%d", decision.action, decision.confidence, len(self.threat_memory))

    @staticmethod
    def _safe_numeric(value: object) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _calibrate_confidence(semantic_confidence: float, signal_score: float, semantic_weight: float = 0.7) -> float:
        """
        Combine semantic and signal-derived scores into a single confidence.
        Default gives more weight to semantic outputs but respects signal evidence.
        Ensures a value in [0.0, 1.0].
        """
        semantic_confidence = max(0.0, min(1.0, float(semantic_confidence)))
        signal_score = max(0.0, min(1.0, float(signal_score)))
        fused = semantic_confidence * semantic_weight + signal_score * (1.0 - semantic_weight)
        # apply a small calibration to avoid exact zeros/ones if borderline
        fused = max(0.0, min(1.0, fused))
        return fused

    # Persistence / Introspection -------------------------------------------

    def export_memory(self, path: str) -> int:
        """
        Export the in-memory threat records to a JSON lines file.
        Returns the number of records written.
        """
        written = 0
        with self._lock:
            records = list(self.threat_memory)
        try:
            with open(path, "w", encoding="utf-8") as fd:
                for rec in records:
                    payload = {
                        "signal": rec.signal,
                        "semantic": rec.semantic,
                        "decision": rec.decision.to_dict(),
                    }
                    fd.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    written += 1
            logger.info("Exported %d threat records to %s", written, path)
        except Exception:
            logger.exception("Failed to export threat memory to %s", path)
            raise
        return written

    def import_memory(self, path: str) -> int:
        """
        Load JSON lines into memory (appends to current memory up to capacity).
        Returns the number of records imported.
        """
        imported = 0
        try:
            with open(path, "r", encoding="utf-8") as fd:
                for line in fd:
                    try:
                        obj = json.loads(line)
                        decision_data = obj.get("decision", {}) or {}
                        decision = Decision(**decision_data)
                        with self._lock:
                            if len(self.threat_memory) < self.threat_memory.maxlen:
                                self.threat_memory.append(ThreatRecord(signal=obj.get("signal", {}), semantic=obj.get("semantic", {}), decision=decision))
                                imported += 1
                            else:
                                # when memory is full, still append to enforce maxlen rollover behavior
                                self.threat_memory.append(ThreatRecord(signal=obj.get("signal", {}), semantic=obj.get("semantic", {}), decision=decision))
                                imported += 1
                    except Exception:
                        logger.exception("Skipping invalid record during import: %s", line.strip()[:200])
            logger.info("Imported %d threat records from %s", imported, path)
        except FileNotFoundError:
            logger.warning("Import path not found: %s", path)
        except Exception:
            logger.exception("Failed to import threat memory from %s", path)
            raise
        return imported

    def clear_memory(self) -> None:
        with self._lock:
            self.threat_memory.clear()
        logger.debug("Cleared threat memory")

    def memory_snapshot(self) -> List[Dict]:
        """Return a serializable snapshot of the current memory (most recent last)."""
        with self._lock:
            return [
                {"signal": r.signal, "semantic": r.semantic, "decision": r.decision.to_dict()}
                for r in list(self.threat_memory)
            ]

    def __len__(self) -> int:
        return len(self.threat_memory)

    def __repr__(self) -> str:
        return f"<DecisionEngine memory_size={len(self.threat_memory)}/{self.threat_memory.maxlen} rules={len(self._rules)}>"

# end of file
