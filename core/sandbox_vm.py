import os
import time 
import json
import shutil
import tempfile
import threading 
from core.neural_engine import NeuralEngine
from utils.logger import log_event, log_error

class SandboxVM: def init(self): self.neural_engine = NeuralEngine() self.temp_dir = tempfile.mkdtemp() self.sandbox_logs = []

def run_in_sandbox(self, file_path):
    try:
        log_event(f"[SANDBOX_VM] Starting sandbox analysis for: {file_path}")

        if not os.path.isfile(file_path):
            raise FileNotFoundError("Target file not found for sandboxing.")

        sandbox_file = shutil.copy(file_path, self.temp_dir)
        result = self._emulate_behavior(sandbox_file)

        prediction = self.neural_engine.analyze_behavior(result)
        verdict = "Malicious" if prediction == 1 else "Benign"

        log_event(f"[SANDBOX_VM] Verdict: {verdict}")
        return {
            "file": file_path,
            "sandbox_result": result,
            "verdict": verdict
        }
    except Exception as e:
        log_error(f"[SANDBOX_VM] Error during sandbox execution: {str(e)}")
        return None

def _emulate_behavior(self, file_path):
    log_event(f"[SANDBOX_VM] Emulating file behavior: {file_path}")
    try:
        # Simulated behavior profiling logic
        time.sleep(2)  # simulate execution delay
        behavior_report = {
            "file_name": os.path.basename(file_path),
            "registry_mods": ["HKCU\\Software\\TestKey"],
            "file_creations": ["tempfile.tmp"],
            "network_activity": ["192.168.1.1:443"],
            "processes_spawned": ["cmd.exe"],
            "anomalies": ["high_cpu_usage"]
        }
        self.sandbox_logs.append(behavior_report)
        return behavior_report
    except Exception as e:
        log_error(f"[SANDBOX_VM] Behavior emulation failed: {str(e)}")
        return {}

def cleanup_sandbox(self):
    try:
        log_event("[SANDBOX_VM] Cleaning up sandbox environment...")
        shutil.rmtree(self.temp_dir)
        self.temp_dir = tempfile.mkdtemp()
    except Exception as e:
        log_error(f"[SANDBOX_VM] Cleanup error: {str(e)}")

def export_sandbox_logs(self, output_path):
    try:
        with open(output_path, 'w') as f:
            json.dump(self.sandbox_logs, f, indent=2)
        log_event(f"[SANDBOX_VM] Sandbox logs exported to: {output_path}")
    except Exception as e:
        log_error(f"[SANDBOX_VM] Failed to export logs: {str(e)}")

