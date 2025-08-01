
import logging
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, Any, Optional, List, Set

_logger = logging.getLogger("sentenialx.agent.trace")
if not _logger.hasHandlers():
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    )
    _handler.setFormatter(_formatter)
    _logger.addHandler(_handler)
_logger.setLevel(logging.INFO)


class AttackEvent:
    """
    Represents a single attack vector event.
    """
    __slots__ = ("event_id", "attack_type", "source", "payload", "timestamp")

    def __init__(
        self,
        event_id: str,
        attack_type: str,
        source: str,
        payload: Any,
        timestamp: Optional[datetime] = None,
    ):
        self.event_id = event_id
        self.attack_type = attack_type
        self.source = source
        self.payload = payload
        self.timestamp = timestamp or datetime.utcnow()

    def __repr__(self):
        return (
            f"<AttackEvent {self.event_id} {self.attack_type} "
            f"from {self.source} at {self.timestamp.isoformat()}>"
        )


class TraceAgent:
    """
    Traces attack vectors and adapts defense strategies on the fly.
    Designed for the agent module of Sentenial-X AI.
    """

    _HIGH_RISK = {"sql_injection", "rce", "xss", "priv_esc"}

    def __init__(self, history_size: int = 1000, threat_threshold: int = 80):
        self._history: deque = deque(maxlen=history_size)
        self._blocked: Set[str] = set()
        self._threat_threshold = threat_threshold
        self._adaptation_log: List[Dict[str, Any]] = []

    def trace(self, event: AttackEvent) -> Dict[str, Any]:
        self._history.append(event)
        risk = self._risk_score(event)
        analysis = {
            "event_id": event.event_id,
            "attack_type": event.attack_type,
            "source": event.source,
            "risk_score": risk,
            "timestamp": event.timestamp.isoformat(),
        }
        _logger.info("Traced event: %s", analysis)
        self._adapt(event, risk)
        return analysis

    def _risk_score(self, event: AttackEvent) -> int:
        if event.attack_type in self._HIGH_RISK:
            return 95
        if "admin" in str(event.payload).lower():
            return 75
        return 45

    def _adapt(self, event: AttackEvent, risk_score: int):
        if risk_score >= self._threat_threshold:
            if event.source not in self._blocked:
                self._blocked.add(event.source)
                self._log_adaptation(
                    action="block_source",
                    source=event.source,
                    reason=f"High risk ({event.attack_type})",
                    event_id=event.event_id,
                )
                _logger.warning("Blocked source: %s", event.source)
        # Dynamic adjustment if attacks spike
        recent = self._recent_events(10)
        high_count = sum(
            1 for evt in recent if self._risk_score(evt) >= self._threat_threshold
        )
        if high_count > 10:
            prev = self._threat_threshold
            self._threat_threshold = min(99, self._threat_threshold + 5)
            self._log_adaptation(
                action="raise_threshold",
                old_threshold=prev,
                new_threshold=self._threat_threshold,
                reason="Attack frequency spike"
            )
            _logger.info("Threat threshold raised to %d", self._threat_threshold)

    def _recent_events(self, minutes: int) -> List[AttackEvent]:
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        return [evt for evt in self._history if evt.timestamp >= cutoff]

    def _log_adaptation(self, **kwargs):
        record = dict(kwargs)
        record["timestamp"] = datetime.utcnow().isoformat()
        self._adaptation_log.append(record)

    @property
    def blocked_sources(self) -> Set[str]:
        return set(self._blocked)

    @property
    def adaptation_log(self) -> List[Dict[str, Any]]:
        return list(self._adaptation_log)


# For live agent module integration, but safe for CLI/unit testing
if __name__ == "__main__":
    agent = TraceAgent(history_size=200, threat_threshold=80)
    events = [
        AttackEvent("evt-001", "sql_injection", "192.168.1.1", "' OR 1=1--"),
        AttackEvent("evt-002", "scan", "192.168.1.2", "masscan"),
        AttackEvent("evt-003", "rce", "192.168.1.1", "os.system('id')"),
        AttackEvent("evt-004", "xss", "192.168.1.3", "<script>alert(1)</script>"),
    ]
    for ev in events:
        result = agent.trace(ev)
        print(result)

    print("Blocked sources:", agent.blocked_sources)
    print("Adaptation log:", agent.adaptation_log)
