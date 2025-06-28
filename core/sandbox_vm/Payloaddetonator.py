import subprocess
import shlex
import time
import logging
import tempfile
import os
from pathlib import Path
from typing import Optional, Dict, Any

LOG_FILE = "payload_detonator.log"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

class PayloadDetonator:
    def __init__(self, sandbox_dir: Optional[str] = None, timeout: int = 60) -> None:
        self.sandbox_dir = Path(sandbox_dir) if sandbox_dir else Path(tempfile.mkdtemp(prefix="payload_sandbox_"))
        self.timeout = timeout
        self._validate_sandbox()
        logging.info(f"Sandbox directory: {self.sandbox_dir}")

    def _validate_sandbox(self) -> None:
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)

    def detonate(self, payload_path: str, args: Optional[str] = None) -> Dict[str, Any]:
        payload = Path(payload_path)
        if not payload.is_file():
            err = f"Payload not found: {payload}"
            logging.error(err)
            raise FileNotFoundError(err)

        cmd = [str(payload.resolve())] + shlex.split(args or "")
        logging.info(f"Detonating payload: {' '.join(cmd)}")

        result = {
            "command": cmd,
            "start_time": None,
            "end_time": None,
            "stdout": "",
            "stderr": "",
            "exit_code": None,
            "timeout": False,
            "exception": None
        }

        start = time.time()
        result["start_time"] = start
        try:
            proc = subprocess.run(
                cmd,
                cwd=self.sandbox_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            end = time.time()
            result.update({
                "end_time": end,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "exit_code": proc.returncode,
            })
            msg = f"Payload executed (Exit code: {proc.returncode})"
            logging.info(msg)
        except subprocess.TimeoutExpired as ex:
            end = time.time()
            result.update({
                "end_time": end,
                "stdout": ex.stdout or "",
                "stderr": ex.stderr or "",
                "timeout": True,
                "exception": f"Timeout ({self.timeout}s): {ex}"
            })
            logging.warning(result["exception"])
        except Exception as ex:
            end = time.time()
            result.update({
                "end_time": end,
                "exception": str(ex)
            })
            logging.error(f"Exception during detonation: {ex}")
        finally:
            self._log_result(result)
        return result

    def _log_result(self, result: Dict[str, Any]) -> None:
        log_lines = [
            "---- Payload Detonation Report ----",
            f"Command   : {' '.join(result['command'])}",
            f"Start     : {result['start_time']}",
            f"End       : {result['end_time']}",
            f"Exit Code : {result['exit_code']}",
            f"Timeout   : {result['timeout']}",
            f"Stdout    : {result['stdout'][:800]}{'...[truncated]' if len(result['stdout']) > 800 else ''}",
            f"Stderr    : {result['stderr'][:800]}{'...[truncated]' if len(result['stderr']) > 800 else ''}",
            f"Exception : {result['exception']}",
            "-----------------------------------"
        ]
        for line in log_lines:
            logging.info(line)

    def cleanup(self) -> None:
        if self.sandbox_dir.exists() and self.sandbox_dir.is_dir():
            for item in self.sandbox_dir.glob("**/*"):
                try:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        item.rmdir()
                except Exception as ex:
                    logging.warning(f"Failed to remove {item}: {ex}")
            try:
                self.sandbox_dir.rmdir()
                logging.info(f"Removed sandbox directory: {self.sandbox_dir}")
            except Exception as ex:
                logging.error(f"Could not remove sandbox dir: {ex}")

if __name__ == "__main__":
    detonator = PayloadDetonator(timeout=30)
    try:
        # Replace with a real payload and arguments for actual usage
        res = detonator.detonate('/path/to/payload', args="--option value")
        print("Detonation Result:", res)
    finally:
        detonator.cleanup()
