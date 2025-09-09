from pathlib import Path
import json, threading, time
from typing import Optional, Callable

from sentenialx_mobile.state import load_state, save_state, bump_learning
from sentenialx_mobile.services.telemetry import bus

REGISTRY = Path.home() / ".sentenialx" / "registry"
MODEL_NAME = "threat_semantics"
CHANNEL = "stable"

class ModelHandle:
    def __init__(self):
        self.version: Optional[str] = None
        self.path: Optional[Path] = None

CURRENT = ModelHandle()

def _read_latest_manifest() -> Optional[dict]:
    latest = REGISTRY / MODEL_NAME / CHANNEL / "latest.json"
    if not latest.exists():
        return None
    try:
        info = json.loads(latest.read_text())
        manifest = Path(info["path"]) / "manifest.json"
        if not manifest.exists():
            return None
        return json.loads(manifest.read_text())
    except Exception:
        return None

def _verify_files(manifest: dict) -> bool:
    import hashlib
    base = Path.home() / ".sentenialx" / "registry" / MODEL_NAME / CHANNEL / manifest["version"]
    for k, f in manifest["files"].items():
        p = base / f["filename"]
        if not p.exists():
            return False
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h != f["sha256"]:
            return False
    return True

def load_student_model(manifest: dict) -> bool:
    # Placeholder: wire to actual ONNX Runtime / Torch loader
    base = Path(manifest["files"]["student"]["filename"]).name
    CURRENT.version = manifest["version"]
    CURRENT.path = (REGISTRY / MODEL_NAME / CHANNEL / manifest["version"] / base)
    return True

def check_and_update_once() -> Optional[str]:
    mf = _read_latest_manifest()
    if not mf:
        return None
    if CURRENT.version == mf.get("version"):
        return None  # already latest
    if not _verify_files(mf):
        return None
    if load_student_model(mf):
        bump_learning(0.01)
        save_state()
        bus.emit("model:update", {"version": mf["version"]})
        return mf["version"]
    return None

def start_models_updater(interval_seconds_getter: Callable[[], int] = lambda: 1800):
    stop_flag = threading.Event()

    def loop():
        while not stop_flag.is_set():
            try:
                newv = check_and_update_once()
                if newv:
                    bus.emit("toast", f"Model updated to {newv}")
            except Exception:
                pass
            # poll at the app's report interval (re-using same getter)
            interval = max(15, int(interval_seconds_getter()))
            for _ in range(interval):
                if stop_flag.is_set():
                    break
                time.sleep(1)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return stop_flag
