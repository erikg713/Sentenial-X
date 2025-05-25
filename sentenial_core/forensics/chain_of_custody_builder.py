
"""
Chain of Custody Builder

Manages secure, immutable, and auditable chains of custody for digital forensics evidence.

Author: [Your Name]
Date: [Today's Date]
"""

import hashlib
import json
import logging
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ChainOfCustodyError(Exception):
    """Custom exception for chain of custody errors."""
    pass


@dataclass(frozen=True)
class ChainOfCustodyEvent:
    """
    Represents a single event in the chain of custody.
    Each event cryptographically links to the previous event for tamper evidence.
    """
    action: str
    actor: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    prev_event_hash: Optional[str] = None
    event_id: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'event_id', self._generate_event_id())

    def _generate_event_id(self) -> str:
        """
        Generates a SHA-256 hash unique to this event, chaining previous event hash for integrity.
        """
        raw = f"{self.timestamp}|{self.action}|{self.actor}|{json.dumps(self.details, sort_keys=True)}|{self.prev_event_hash or ''}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # Placeholder for digital signature (extend with cryptography library if needed)
    def sign_event(self, private_key: Any = None) -> str:
        """
        Digitally sign the event.
        Replace with actual cryptographic signing as required.
        """
        logger.debug("Digital signing placeholder called. Integrate your cryptographic library here.")
        return "SIGNATURE_PLACEHOLDER"


class ChainOfCustody:
    """
    Immutable chain of custody for a digital evidence item.
    Each event is cryptographically linked to its predecessor.
    """
    def __init__(self, evidence_id: str, created_by: str):
        self._evidence_id = evidence_id
        self._created_by = created_by
        self._created_at = datetime.utcnow().isoformat() + "Z"
        self._events: List[ChainOfCustodyEvent] = []
        logger.info(f"Initialized chain of custody for evidence: {evidence_id}")

    @property
    def evidence_id(self) -> str:
        return self._evidence_id

    @property
    def created_by(self) -> str:
        return self._created_by

    @property
    def created_at(self) -> str:
        return self._created_at

    @property
    def events(self) -> List[ChainOfCustodyEvent]:
        # Return a deep copy to preserve immutability
        return deepcopy(self._events)

    def add_event(self, action: str, actor: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an immutable event to the chain, cryptographically linked to the previous event.
        """
        prev_hash = self._events[-1].event_id if self._events else None
        event = ChainOfCustodyEvent(
            action=action,
            actor=actor,
            details=details or {},
            prev_event_hash=prev_hash,
        )
        self._events.append(event)
        logger.info(f"Event added: {action} by {actor} at {event.timestamp

