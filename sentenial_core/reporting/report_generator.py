import os
import json
import logging
from datetime import datetime
from typing import Any, Dict

REPORT_DIR = "data/reports/generated"
SUBDIRS = ["threats", "exploits", "sandbox", "telemetry", "ransomware"]

# Ensure directory structure
for sub in SUBDIRS:
    os.makedirs(os.path.join(REPORT_DIR, sub), exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ReportGenerator")

class ReportGenerator:
    def __init__(self):
        self.report_dir = REPORT_DIR

    def _current_timestamp(self) -> str:
        return datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")

    def _write_file(self, path: str, content: str, mode: str = "w") -> None:
        with open(path, mode, encoding="utf-8") as f:
            f.write(content)
        logger.info(f"[✓] Report written to {path}")

    def generate_threat_report(self, data: Dict[str, Any]) -> str:
        filename = f"threat_report_{self._current_timestamp()}.json"
        full_path = os.path.join(self.report_dir, "threats", filename)
        self._write_file(full_path, json.dumps(data, indent=4))
        return full_path

    def generate_exploit_log(self, module_name: str, log_output: str) -> str:
        filename = f"exploit_{module_name}_{self._current_timestamp()}.log"
        full_path = os.path.join(self.report_dir, "exploits", filename)
        self._write_file(full_path, log_output)
        return full_path

    def generate_ransomware_emulation(self, emulation_data: Dict[str, Any]) -> str:
        filename = f"ransomware_emulation_{self._current_timestamp()}.json"
        full_path = os.path.join(self.report_dir, "ransomware", filename)
        self._write_file(full_path, json.dumps(emulation_data, indent=4))
        return full_path

    def generate_sandbox_snapshot(self, snapshot_data: Dict[str, Any]) -> str:
        filename = f"sandbox_snapshot_{self._current_timestamp()}.yaml"
        full_path = os.path.join(self.report_dir, "sandbox", filename)
        try:
            import yaml
            yaml_content = yaml.dump(snapshot_data)
            self._write_file(full_path, yaml_content)
        except ImportError:
            logger.error("PyYAML not installed: `pip install pyyaml`")
        return full_path

    def generate_telemetry_stream(self, entries: list[Dict[str, Any]]) -> str:
        filename = f"telemetry_{self._current_timestamp()}.ndjson"
        full_path = os.path.join(self.report_dir, "telemetry", filename)
        with open(full_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        logger.info(f"[✓] Telemetry stream saved: {filename}")
        return full_path

if __name__ == "__main__":
    rg = ReportGenerator()

    # Simulated threat report
    rg.generate_threat_report({
        "threat_level": "HIGH",
        "source": "process_inspector",
        "details": {
            "pid": 1337,
            "anomaly_score": 0.92,
            "signature": "C2_Backdoor_ABC"
        }
    })

    # Simulated exploit log
    rg.generate_exploit_log("ms17_010_eternalblue", "[*] Exploit successful. Shell opened.")

    # Ransomware emulation result
    rg.generate_ransomware_emulation({
        "files_encrypted": 157,
        "ransom_note_dropped": True,
        "payment_address": "pi_network:sentenialx"
    })

    # Sandbox snapshot
    rg.generate_sandbox_snapshot({
        "processes": ["svchost.exe", "explorer.exe"],
        "open_ports": [80, 443],
        "registry": {"HKLM\\Software\\Evil": "Present"}
    })

    # Streaming telemetry
    rg.generate_telemetry_stream([
        {"time": "2025-07-16T01:00:00Z", "cpu": 38, "mem": 63},
        {"time": "2025-07-16T01:01:00Z", "cpu": 41, "mem": 65},
    ])