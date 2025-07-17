import os
import tempfile
import shutil
import logging
import threading
from pathlib import Path
from typing import List

from apps.ransomeware_emulator.encryption import encrypt_file, decrypt_file
from sentenialx.ai_core.logger import log_threat_event
from apps.ransomeware_emulator.monitor import SandboxMonitor  # Optional, must exist

logger = logging.getLogger("RansomwareSandbox")
logging.basicConfig(level=logging.INFO)

class RansomwareSandbox:
    def __init__(self, payload_func, monitor: bool = False):
        self.payload_func = payload_func
        self.monitor_enabled = monitor
        self.sandbox_root = Path(tempfile.mkdtemp(prefix="sentenial_sandbox_"))
        self.monitor = SandboxMonitor(self.sandbox_root) if monitor else None
        self.lock = threading.Lock()
        logger.info(f"Sandbox initialized at {self.sandbox_root}")

    def setup_test_environment(self, file_count: int = 5):
        logger.info("Setting up test environment with sample files...")
        for i in range(file_count):
            file_path = self.sandbox_root / f"testfile_{i}.txt"
            with open(file_path, "w") as f:
                f.write(f"This is test file #{i}.\n" * 10)
        logger.info(f"Created {file_count} test files.")

    def run_payload(self):
        logger.info("Executing ransomware payload in sandbox...")
        try:
            if self.monitor:
                self.monitor.start()

            with self.lock:
                self.payload_func(self.sandbox_root)

        except Exception as e:
            logger.error(f"Error during payload execution: {e}")
        finally:
            if self.monitor:
                self.monitor.stop()
                self.monitor.report()

    def cleanup(self):
        logger.info("Cleaning up sandbox environment...")
        try:
            shutil.rmtree(self.sandbox_root)
            logger.info(f"Deleted sandbox at {self.sandbox_root}")
        except Exception as e:
            logger.error(f"Failed to clean up sandbox: {e}")

    def list_files(self) -> List[Path]:
        return list(self.sandbox_root.glob("*"))

    def restore_original_files(self):
        logger.info("Restoring sandbox files to original state...")
        try:
            for file in self.list_files():
                if file.suffix == ".enc":
                    decrypt_file(file, file.with_suffix(".txt"), "beastmodepassword")
        except Exception as e:
            logger.error(f"Restoration failed: {e}")

# 🚀 PAYLOAD: Sample encryption + threat logging
def sample_encrypt_payload(root: Path):
    logger.info("Running ransomware simulation...")
    for file_path in root.glob("*.txt"):
        encrypted_path = file_path.with_suffix(".enc")
        encrypt_file(file_path, encrypted_path, "beastmodepassword")
        os.remove(file_path)
        logger.debug(f"Encrypted {file_path.name}")

        # 🔥 Log threat
        log_threat_event(
            threat_type="ransomware_activity",
            source="sandbox",
            payload=f"Encrypted {file_path.name}",
            confidence=0.98
        )

# 🎯 ENTRYPOINT
if __name__ == "__main__":
    sandbox = RansomwareSandbox(payload_func=sample_encrypt_payload, monitor=True)
    sandbox.setup_test_environment()
    sandbox.run_payload()
    sandbox.restore_original_files()
    sandbox.cleanup()
