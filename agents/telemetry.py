# sentenial-x/agents/telemetry.py
import time
import json
from typing import List, Dict, Optional
from .config import LOG_BATCH_SIZE, LOG_FLUSH_INTERVAL, ENABLE_ENCRYPTION, ENCRYPTION_KEY

# Optional: simple symmetric encryption (for demonstration)
from cryptography.fernet import Fernet

# Generate key from ENCRYPTION_KEY (must be 32-byte base64)
def get_fernet_key(key_str: str) -> bytes:
    import base64, hashlib
    hash_key = hashlib.sha256(key_str.encode()).digest()
    return base64.urlsafe_b64encode(hash_key)

FERNET = Fernet(get_fernet_key(ENCRYPTION_KEY)) if ENABLE_ENCRYPTION else None


class TelemetryBuffer:
    """
    Buffers agent logs and telemetry data.
    Sends logs in batches or periodically.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.buffer: List[Dict] = []
        self.last_flush_time = time.time()

    def add_log(self, log: str, meta: Optional[Dict] = None):
        entry = {
            "timestamp": time.time(),
            "agent_id": self.agent_id,
            "log": log,
            "meta": meta or {}
        }
        self.buffer.append(entry)

        # Flush if batch full or flush interval reached
        if len(self.buffer) >= LOG_BATCH_SIZE or (time.time() - self.last_flush_time) >= LOG_FLUSH_INTERVAL:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        payload = json.dumps(self.buffer)
        if ENABLE_ENCRYPTION and FERNET:
            payload = FERNET.encrypt(payload.encode()).decode()

        # In production, send to orchestrator via REST/gRPC
        self._send_to_orchestrator(payload)

        self.buffer.clear()
        self.last_flush_time = time.time()

    def _send_to_orchestrator(self, payload: str):
        # Placeholder for actual network call
        print(f"[Telemetry] Sending {len(payload)} bytes to orchestrator for agent {self.agent_id}")
        # Example: requests.post("https://orchestrator/telemetry", data=payload)
