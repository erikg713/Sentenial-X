from pathlib import Path
import json

def test_pipeline_prod_execution():
    from scripts.run_pipeline_prod import run_pipeline_prod
    run_pipeline_prod(sequential=True)

    # Check telemetry folder
    telemetry_dir = Path("telemetry")
    assert telemetry_dir.exists()
    json_files = list(telemetry_dir.glob("telemetry_*.json"))
    assert len(json_files) > 0

    # Validate JSON structure
    for file in json_files:
        with file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            assert isinstance(data, dict)
