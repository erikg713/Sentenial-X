import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from executor import execute_command
from memory import remember
from ml_classifier import classify_command, train_model

# Default location for the commands queue
COMMAND_QUEUE = Path("commands") / "agent_commands.json"

# Configure a module logger so the rest of the application can control output
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Basic configuration when logger not configured by the application
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class FileLockError(Exception):
    pass


def _acquire_lock(lock_path: Path, retries: int = 10, delay: float = 0.1, stale_after: float = 30.0) -> None:
    """
    Obtain a simple filesystem lock by creating a lock file exclusively.
    If a stale lock exists (old enough), remove it.
    Raises FileLockError if lock cannot be acquired within retries.
    """
    for attempt in range(retries):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(f"{os.getpid()}\n{time.time()}\n")
            return
        except FileExistsError:
            try:
                stat = lock_path.stat()
                age = time.time() - stat.st_mtime
                if age > stale_after:
                    logger.warning("Removing stale lock file %s (age %.1fs)", lock_path, age)
                    lock_path.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                # race condition: file disappeared between existence check and stat
                continue
            time.sleep(delay)
    raise FileLockError(f"Could not acquire lock {lock_path} after {retries} retries")


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink(missing_ok=True)
    except Exception as exc:
        logger.debug("Failed to remove lock file %s: %s", lock_path, exc)


def _atomic_write_json(path: Path, data: Any, **dump_kwargs: Any) -> None:
    """
    Write JSON to a temp file in the same directory and atomically replace the target.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, **dump_kwargs)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def listen_for_commands(
    agent_id: str,
    queue_path: Optional[Path] = None,
    dry_run: bool = False,
    lock_retries: int = 10,
    lock_delay: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Read queued commands for `agent_id` from the JSON queue, execute them,
    remember them, and return a list of result dicts for commands processed.

    - queue_path: optional override for the commands JSON file path.
    - dry_run: if True, commands will be classified and logged but not executed.
    - Returns a list of processed command result objects.
    """
    queue = Path(queue_path) if queue_path else COMMAND_QUEUE
    results: List[Dict[str, Any]] = []

    if not queue.exists():
        logger.debug("Command queue does not exist: %s", queue)
        return results

    lock_path = queue.with_suffix(queue.suffix + ".lock") if queue.suffix else queue.with_name(queue.name + ".lock")
    try:
        _acquire_lock(lock_path, retries=lock_retries, delay=lock_delay)
    except FileLockError as e:
        logger.error("Unable to acquire lock for queue: %s", e)
        return results

    try:
        # Read and validate JSON
        try:
            with queue.open("r", encoding="utf-8") as f:
                raw = f.read()
                if not raw.strip():
                    commands_data: Dict[str, Any] = {}
                else:
                    commands_data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error("Failed to decode JSON from %s: %s", queue, e)
            return results
        except FileNotFoundError:
            logger.debug("Queue file disappeared while trying to open it: %s", queue)
            return results

        if not isinstance(commands_data, dict):
            logger.error("Unexpected queue format: expected JSON object at root")
            return results

        agent_commands = commands_data.get(agent_id)
        if not agent_commands:
            logger.debug("No commands for agent '%s' in queue", agent_id)
            return results

        if not isinstance(agent_commands, list):
            logger.error("Commands for agent '%s' must be a list, got %s", agent_id, type(agent_commands).__name__)
            return results

        logger.info("Found %d command(s) for agent '%s'", len(agent_commands), agent_id)

        for cmd in agent_commands:
            cmd_result: Dict[str, Any] = {"agent_id": agent_id, "command": cmd, "executed": False, "error": None, "label": None}
            try:
                label = classify_command(cmd)
                cmd_result["label"] = label
                logger.info("[ML] Command classified as: %s -- %s", label, cmd)
            except Exception as e:
                # Classification failure shouldn't stop command execution
                logger.debug("Classification failed for command %r: %s", cmd, e)

            if dry_run:
                logger.info("[DRY RUN] Skipping execution of: %s", cmd)
                cmd_result["executed"] = False
            else:
                try:
                    logger.info("[EXECUTE] %s", cmd)
                    exec_result = execute_command(cmd)
                    cmd_result["executed"] = True
                    cmd_result["result"] = exec_result
                    logger.debug("Execution result for %r: %r", cmd, exec_result)
                except Exception as e:
                    logger.exception("Execution failed for command %r", cmd)
                    cmd_result["error"] = str(e)

            # Persist memory about the command (best-effort)
            try:
                remember_record = {
                    "source": "command_queue",
                    "agent_id": agent_id,
                    "command": cmd,
                    "label": cmd_result.get("label"),
                    "executed": cmd_result.get("executed"),
                    "timestamp": time.time(),
                }
                remember(remember_record)
            except Exception:
                logger.debug("Failed to remember command %r", cmd, exc_info=True)

            results.append(cmd_result)

        # Remove processed commands for this agent and write back atomically
        try:
            # Make a shallow copy so we don't mutate original structure that may be shared
            new_data = dict(commands_data)
            if agent_id in new_data:
                del new_data[agent_id]
            _atomic_write_json(queue, new_data, indent=2, ensure_ascii=False, sort_keys=True)
            logger.info("Removed %d command(s) for agent '%s' from queue", len(agent_commands), agent_id)
        except Exception:
            logger.exception("Failed to update commands queue file after processing")

        # Train the model in background so processing is non-blocking for callers.
        def _background_train():
            try:
                logger.info("Starting background model training")
                train_model()
                logger.info("Background model training finished")
            except Exception:
                logger.exception("Background model training failed")

        threading.Thread(target=_background_train, name="ml-train-thread", daemon=True).start()

    finally:
        # Ensure lock removed
        _release_lock(lock_path)

    return results
