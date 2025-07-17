import logging
from pathlib import Path
from typing import Callable, Dict, Any
from datetime import datetime

from .sandbox import RansomwareSandbox
from .payloads import PAYLOAD_REGISTRY

logger = logging.getLogger("RansomwareEmulator")
logging.basicConfig(level=logging.INFO)


class RansomwareEmulator:
    def __init__(self):
        """
        Central controller for ransomware emulation campaigns.
        """
        self.results = []

    def run_campaign(self, payload_name: str, monitor: bool = True, file_count: int = 5) -> Dict[str, Any]:
        """
        Launch a sandboxed ransomware simulation.

        Args:
            payload_name (str): The key name of the payload function to run.
            monitor (bool): Whether to enable filesystem monitoring.
            file_count (int): Number of dummy files to generate.

        Returns:
            Dict[str, Any]: Summary of the campaign results.
        """
        if payload_name not in PAYLOAD_REGISTRY:
            raise ValueError(f"Unknown payload: {payload_name}")

        logger.info(f"Launching ransomware campaign: {payload_name}")
        payload_func: Callable = PAYLOAD_REGISTRY[payload_name]
        sandbox = RansomwareSandbox(payload_func=payload_func, monitor=monitor)

        sandbox.setup_test_environment(file_count=file_count)
        sandbox.run_payload()

        files_after = sandbox.list_files()
        report = {
            "payload": payload_name,
            "timestamp": datetime.utcnow().isoformat(),
            "sandbox_path": str(sandbox.sandbox_root),
            "file_count": len(files_after),
            "encrypted_files": [f.name for f in files_after if f.suffix == ".enc"]
        }

        logger.info(f"Campaign '{payload_name}' completed with {len(report['encrypted_files'])} encrypted files.")
        self.results.append(report)

        # Optional: sandbox.restore_original_files()
        sandbox.cleanup()

        return report

    def list_payloads(self) -> Dict[str, Callable]:
        """
        List all registered payloads.
        """
        return PAYLOAD_REGISTRY

    def get_results(self) -> list:
        """
        Get all campaign results run in this session.
        """
        return self.results


# CLI/Test Execution
if __name__ == "__main__":
    emulator = RansomwareEmulator()
    available = emulator.list_payloads()

    print("[*] Available Payloads:")
    for name in available:
        print(f" - {name}")

    print("\n[*] Running default campaign...")
    result = emulator.run_campaign("basic_encrypt", monitor=True)
    print("\n[*] Campaign Result:")
    for k, v in result.items():
        print(f"{k}: {v}")
