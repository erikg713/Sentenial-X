# sentenialx/models/orchestrator/orchestrate.py
import argparse
from services.cortex.cli import train  # Import or subprocess
from sentenialx.models.artifacts import register_artifact

def package_cortex():
    # Trigger training or load
    # ... (training logic)
    artifact_path = Path("sentenialx/models/artifacts/distill/threat_student_v1/pytorch_model.bin")
    register_artifact("distill", artifact_path, "1.0.0")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["package"])
    parser.add_argument("--component", default="cortex")
    args = parser.parse_args()
    if args.stage == "package" and args.component == "cortex":
        package_cortex()
# sentenialx/models/orchestrator/orchestrate.py (excerpt)
from sentenialx.models.lora.lora_tuner import tune_lora

def tune_component(component: str):
    if component == "lora":
        tune_lora(dataset_path="sentenialx/data/processed/new_threats.csv")

# CLI
parser.add_argument("--stage", choices=["tune", "package"])
parser.add_argument("--component", default="lora")
if args.stage == "tune":
    tune_component(args.component)
