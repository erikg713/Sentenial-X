# ===== File: core/detection/base.py =====
from __future__ import annotations
import dataclasses
import enum
import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol


class Severity(enum.IntEnum):
    INFO = 10
    LOW = 20
    MEDIUM = 30
    HIGH = 40
    CRITICAL = 50


@dataclass(frozen=True)
class DetectionVerdict:
    score: float
    severity: Severity
    label: str
    reason: str
    meta: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "severity": int(self.severity),
            "label": self.label,
            "reason": self.reason,
            "meta": dict(self.meta),
        }


@dataclass(frozen=True)
class DetectionEvent:
    kind: str
    subject: str
    payload: Any
    attributes: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    ts: float = dataclasses.field(default_factory=lambda: time.time())
    id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)

    def stable_key(self) -> str:
        h = hashlib.sha256()
        h.update(self.kind.encode())
        h.update(self.subject.encode())
        payload_bytes = self.payload if isinstance(self.payload, (bytes, bytearray)) else json.dumps(self.payload, sort_keys=True, default=str).encode()
        h.update(payload_bytes)
        return h.hexdigest()


class Detector(Protocol):
    name: str

    def detect(self, event: DetectionEvent) -> Optional[DetectionVerdict]:
        ...
      
