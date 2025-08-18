"""
Encrypted Logs Module for Sentenial-X
-------------------------------------
Handles secure writing and retrieval of logs using symmetric encryption.
"""

import os
import time
import json
from cryptography.fernet import Fernet
from typing import Optional, List, Dict

# ---------------- Configuration ----------------
LOG_DIR = os.getenv("ENCRYPTED_LOG_DIR", "data/logs/encrypted")
KEY_PATH = os.getenv("FERNET_KEY_PATH", "data/logs/fernet.key")

os.makedirs(LOG_DIR, exist_ok=True)


def get_fernet_key(key_path: str = KEY_PATH) -> bytes:
    """
    Load or generate a Fernet key for encryption.
    """
    if os.path.exists(key_path):
        return open(key_path, "rb").read()
    key = Fernet.generate_key()
    with open(key_path, "wb") as f:
        f.write(key)
    return key


FERNET = Fernet(get_fernet_key())


# ---------------- Log Entry Dataclass ----------------
class EncryptedLogEntry:
    def __init__(self, timestamp: float, level: str, message: str, meta: Optional[Dict] = None):
        self.timestamp = timestamp
        self.level = level
        self.message = message
        self.meta = meta or {}

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "level": self.level,
            "message": self.message,
            "meta": self.meta
        }

    @staticmethod
    def from_encrypted(data: bytes, fernet: Fernet):
        decrypted = fernet.decrypt(data)
        obj = json.loads(decrypted.decode())
        return EncryptedLogEntry(**obj)


# ---------------- Log Manager ----------------
class EncryptedLogManager:
    """
    Manages encrypted log files.
    """

    def __init__(self, log_dir: str = LOG_DIR, fernet: Fernet = FERNET):
        self.log_dir = log_dir
        self.fernet = fernet

    def write_log(self, entry: EncryptedLogEntry):
        """
        Encrypt and write a log entry to a daily file.
        """
        filename = time.strftime("logs_%Y_%m_%d.log", time.gmtime())
        path = os.path.join(self.log_dir, filename)
        data = json.dumps(entry.to_dict()).encode()
        encrypted = self.fernet.encrypt(data)
        with open(path, "ab") as f:
            f.write(encrypted + b"\n")

    def read_logs(self, date: Optional[str] = None) -> List[EncryptedLogEntry]:
        """
        Read and decrypt logs for the given date (YYYY_MM_DD), default to today.
        """
        date_str = date or time.strftime("%Y_%m_%d", time.gmtime())
        filename = f"logs_{date_str}.log"
        path = os.path.join(self.log_dir, filename)
        if not os.path.exists(path):
            return []
        entries = []
        with open(path, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = EncryptedLogEntry.from_encrypted(line, self.fernet)
                    entries.append(entry)
                except Exception:
                    continue
        return entries
