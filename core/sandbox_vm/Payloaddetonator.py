import subprocess
import shlex
import time
import logging
import tempfile
import os
from typing import Optional, Dict, Any

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("payload_detonator.log"),
        logging.StreamHandler()
    ]
)

class PayloadDetonator:
    def __init__(self, sandbox_dir: Optional[str] = None, timeout: int = 60):
        self.sandbox_dir = sandbox_dir or tempfile.mkdtemp(prefix="payload_sandbox_")
        self.timeout = timeout
        logging.info(f"Sandbox directory set to: {self.sandbox_dir}")

    def detonate(self, payload_path: str, args: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes the given payload inside the sandbox directory and captures its output.

        Args:
            payload_path (str): Path to the payload executable/script.
            args (Optional[str]): Additional arguments for the payload.

        Returns:
            Dict[str, Any]: Dictionary containing execution details.
        """
        if not os.path.isfile(payload_path):
            logging.error(f"Payload not found: {payload_path}")
            raise FileNotFoundError(f"Payload not found: {payload_path}")

        command = [payload_path] + shlex.split(args or "")
        logging.info(f"Detonating payload: {' '.join(command)}")

        result = {
            "command": command,
            "start_time": None,
            "end_time": None,
            "stdout": "",
            "stderr": "",
            "exit_code": None,
            "timeout": False,
            "exception": None
        }

        try:
            result["start_time"] = time.time()
            proc = subprocess.run(
                command,
                cwd=self.sandbox_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            result["end_time"] = time.time()
            result["stdout"] = proc.stdout
            result["stderr"] = proc.stderr
            result["exit_code"] = proc.returncode
            logging.info(f"Payload executed. Exit code: {proc.returncode}")
        except subprocess.TimeoutExpired as ex:
            result["end_time"] = time.time()
            result["stdout"] = ex.stdout or ""
            result["stderr"] = ex.stderr or ""
            result["timeout"] = True
            result["exception"] = str(ex)
            logging.warning(f"Payload timed out after {self.timeout} seconds.")
        except Exception as ex:
            result["end_time"] = time.time()
            result["exception"] = str(ex)
            logging.error(f"Exception during payload detonation: {ex}")

        # Optionally, log result details
        self._log_result(result)
        return result

    def _log_result(self, result: Dict[str, Any]) -> None:
        log_msg = (
            f"Command: {' '.join(result['command'])}\n"
            f"Start: {result['start_time']}\n"
            f"End: {result['end_time']}\n"
            f"Exit Code: {result['exit_code']}\n"
            f"Timeout: {result['timeout']}\n"
            f"Stdout: {result['stdout'][:500]}{'...' if len(result['stdout']) > 500 else ''}\n"
            f"Stderr: {result['stderr'][:500]}{'...' if len(result['stderr']) > 500 else ''}\n"
            f"Exception: {result['exception']}\n"
            "----------------------------------------"
        )
        logging.info(log_msg)

    def cleanup(self):
        """ Removes the sandbox directory and its contents. """
        if os.path.exists(self.sandbox_dir):
            try:
                for root, dirs, files in os.walk(self.sandbox_dir, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(self.sandbox_dir)
                logging.info(f"Cleaned up sandbox directory: {self.sandbox_dir}")
            except Exception as ex:
                logging.error(f"Failed to clean up sandbox: {ex}")

# Example usage:
if __name__ == "__main__":
    detonator = PayloadDetonator(timeout=30)
    try:
        result = detonator.detonate('/path/to/payload', args="--option value")
        print("Detonation Result:", result)
    finally:
        detonator.cleanup()
