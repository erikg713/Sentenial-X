import os
import tempfile
import shutil
import logging
import threading
from pathlib import Path
from typing import List

from .encryption import encrypt_file, decrypt_file  # Assumed helper methods
from .monitor import SandboxMonitor  # Optional module for file access telemetry
# apps/ransomeware_emulator/sandbox.py

from apps.ransomeware_emulator.encryption import encrypt_file, decrypt_file
from sentenialx.ai_core.logger import log_threat_event  # <<< add this

# apps/ransomeware_emulator/sandbox.py

from apps.ransomeware_emulator.encryption import encrypt_file, decrypt_file
from sentenialx.ai_core.logger import log_threat_event  # <<< add this

def main():
    test_file = "test.txt"
    enc_file = "test.enc"
    password = "beastmodepassword"

    encrypt_file(test_file, enc_file, password)

    # 🔥 Log encryption as a threat
    log_threat_event(
        threat_type="ransomware_activity",
        source="sandbox",
        payload=f"Encrypted {test_file} -> {enc_file}",
        confidence=0.98
    )

    decrypt_file(enc_file, "test_decrypted.txt", password)

if __name__ == "__main__":
    main()


def main():
    test_file = "test.txt"
    enc_file = "test.enc"
    password = "beastmodepassword"

    encrypt_file(test_file, enc_file, password)

    # 🔥 Log encryption as a threat
    log_threat_event(
        threat_type="ransomware_activity",
        source="sandbox",
        payload=f"Encrypted {test_file} -> {enc_file}",
        confidence=0.98
    )

    decrypt_file(enc_file, "test_decrypted.txt", password)

if __name__ == "__main__":
    main()

logger = logging.getLogger("RansomwareSandbox")
logging.basicConfig(level=logging.INFO)


class RansomwareSandbox:
    def __init__(self, payload_func, monitor: bool = False):
        """
        Initializes a ransomware simulation sandbox.
        
        Args:
            payload_func: A function that simulates ransomware behavior, e.g., file encryption.
            monitor: Whether to enable file telemetry monitoring.
        """
        self.payload_func = payload_func
        self.monitor_enabled = monitor
        self.sandbox_root = Path(tempfile.mkdtemp(prefix="sentenial_sandbox_"))
        self.monitor = SandboxMonitor(self.sandbox_root) if monitor else None
        self.lock = threading.Lock()
        logger.info(f"Sandbox initialized at {self.sandbox_root}")

    def setup_test_environment(self, file_count: int = 10):
        """
        Creates test files to simulate real user data for ransomware to target.
        """
        logger.info("Setting up test environment with sample files...")
        for i in range(file_count):
            file_path = self.sandbox_root / f"testfile_{i}.txt"
            with open(file_path, "w") as f:
                f.write(f"This is test file #{i}.\n" * 10)
        logger.info(f"Created {file_count} test files.")

    def run_payload(self):
        """
        Runs the simulated ransomware payload inside the sandbox.
        """
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
        """
        Cleans up the sandbox environment.
        """
        logger.info("Cleaning up sandbox environment...")
        try:
            shutil.rmtree(self.sandbox_root)
            logger.info(f"Deleted sandbox at {self.sandbox_root}")
        except Exception as e:
            logger.error(f"Failed to clean up sandbox: {e}")

    def list_files(self) -> List[Path]:
        """
        Returns a list of files in the sandbox.
        """
        return list(self.sandbox_root.glob("*"))

    def restore_original_files(self):
        """
        Decrypts or resets test files if reversible ransomware simulation was used.
        """
        logger.info("Restoring sandbox files to original state...")
        try:
            for file in self.list_files():
                if file.suffix == ".enc":
                    decrypt_file(file)
        except Exception as e:
            logger.error(f"Restoration failed: {e}")


# Example Payload Function (for demonstration only)
def sample_encrypt_payload(root: Path):
    """
    Simulated ransomware payload that encrypts all .txt files in a directory.
    """
    logger.info("Running sample ransomware encryption payload...")
    for file_path in root.glob("*.txt"):
        encrypt_file(file_path)  # assumed method that creates file_path + ".enc"
        os.remove(file_path)
        logger.debug(f"Encrypted {file_path.name}")


# If run standalone
if __name__ == "__main__":
    sandbox = RansomwareSandbox(payload_func=sample_encrypt_payload, monitor=True)
    sandbox.setup_test_environment(file_count=5)
    sandbox.run_payload()
    sandbox.restore_original_files()
    sandbox.cleanup()
