# sentenialx/models/orchestrator/orchestrate.py
import argparse, json, time
from pathlib import Path
import yaml

from sentenialx.models.encoder.text_encoder import ThreatTextEncoder
from sentenialx.models.lora.lora_tuner import lora_finetune
from sentenialx.models.distill.distill_trainer import distill_to_student
from sentenialx.models.orchestrator.versioning import SemVer
from sentenialx.models.orchestrator.registry import push_artifact

def load_cfg(cfg_path: Path):
    return yaml.safe_load(Path(cfg_path).read_text())

def read_prev_version(reg_root: Path, model_name: str, channel: str) -> str:
    latest = reg_root.expanduser() / model_name / channel / "latest.json"
    if latest.exists():
        try:
            return json.loads(latest.read_text())["version"]
        except Exception:
            pass
    return "0.0.0"

def main():
    p = argparse.ArgumentParser(description="Sentenial-X Orchestrator")
    p.add_argument("--config", "-c", default=str(Path(__file__).with_name("config.yaml")))
    p.add_argument("--stage", choices=["all", "prep", "lora", "distill", "package"], default="all")
    args = p.parse_args()

    cfg = load_cfg(Path(args.config))
    model_name = cfg["model_name"]
    artifacts_root = Path(cfg["artifacts_root"])
    registry_root = Path(cfg["registry_root"]).expanduser()
    channel = cfg["versioning"]["channel"]
    strategy = cfg["versioning"]["strategy"]

    teacher_base = Path(cfg["teacher_base_model"])
    data_dir = Path(cfg["data_dir"])

    # 0) prep
    prep_dir = artifacts_root / "prep"
    prep_dir.mkdir(parents=True, exist_ok=True)
    (prep_dir / "ok.txt").write_text("dataset checked")
    # (real world: validate datasets, tokenize, split, etc.)

    if args.stage in ("prep",):
        print("Prep complete.")
        return

    # 1) LoRA fine-tune
    lora_dir = artifacts_root / "lora"
    lora_weights = lora_dir / "lora_weights.bin"
    if cfg["lora"]["enabled"] and args.stage in ("all", "lora", "package", "distill"):
        lora_weights = lora_finetune(teacher_base, data_dir, lora_dir)
        print(f"LoRA done: {lora_weights}")

    # 2) Distill to student
    distill_dir = artifacts_root / "distill"
    student_model = distill_dir / "student_model.onnx"
    if cfg["distill"]["enabled"] and args.stage in ("all", "distill", "package"):
        student_model = distill_to_student(teacher_base, lora_weights, data_dir, distill_dir)
        print(f"Student built: {student_model}")

    # 3) Package + push to registry
    prev = read_prev_version(registry_root, model_name, channel)
    new_version = str(SemVer.parse(prev).bump(strategy))
    print(f"Publishing version {new_version} (prev {prev}) on channel {channel}")

    files = {
        "student": student_model,
        "lora": lora_weights,
        "teacher_ref": teacher_base,
    }
    model_dir = push_artifact(registry_root, model_name, new_version, channel, files)
    print(f"Pushed build to registry: {model_dir}")

    # 4) rollout plan (placeholder)
    (model_dir / "rollout.json").write_text(json.dumps({
        "channel": channel,
        "version": new_version,
        "staged": True,
        "rings": [
            {"name": "canary", "percent": 5},
            {"name": "beta", "percent": 20},
            {"name": "stable", "percent": 100}
        ],
        "created_at": time.time()
    }, indent=2))
    print("Rollout plan staged.")

if __name__ == "__main__":
    main()
