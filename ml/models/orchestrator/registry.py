# sentenialx/models/orchestrator/registry.py
from pathlib import Path
import json, time, shutil, hashlib
from typing import Optional

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def push_artifact(registry_root: Path, model_name: str, version: str, channel: str, files: dict) -> Path:
    """
    files: dict logical_name -> Path
    """
    base = ensure_dir(registry_root.expanduser())
    model_dir = ensure_dir(base / model_name / channel / version)
    manifest = {
        "model": model_name,
        "channel": channel,
        "version": version,
        "created_at": time.time(),
        "files": {},
    }
    for key, src in files.items():
        dst = model_dir / src.name
        shutil.copy2(src, dst)
        manifest["files"][key] = {
            "filename": dst.name,
            "sha256": sha256(dst),
        }
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # update latest pointer
    (base / model_name / channel / "latest.json").write_text(json.dumps({
        "version": version,
        "path": str(model_dir),
        "created_at": time.time(),
    }, indent=2))
    return model_dir

def pull_latest_manifest(registry_root: Path, model_name: str, channel: str) -> Optional[dict]:
    latest = (Path(registry_root).expanduser() / model_name / channel / "latest.json")
    if not latest.exists():
        return None
    data = json.loads(latest.read_text())
    manifest_path = Path(data["path"]) / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text())
