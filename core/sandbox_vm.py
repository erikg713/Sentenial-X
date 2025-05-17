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

import os import subprocess import shutil import time from datetime import datetime from utils.logger import log_event from utils.helpers import generate_session_id

class SandboxVM: def init(self, vm_path="/opt/sandbox_vm/", snapshot_name="CleanState"): self.vm_path = vm_path self.snapshot_name = snapshot_name self.session_id = generate_session_id()

def _log(self, msg):
    log_event("SandboxVM", msg, session_id=self.session_id)

def restore_snapshot(self):
    self._log(f"Restoring snapshot '{self.snapshot_name}'")
    try:
        subprocess.run(["vboxmanage", "snapshot", self.vm_path, "restore", self.snapshot_name], check=True)
        self._log("Snapshot restored successfully.")
        return True
    except subprocess.CalledProcessError as e:
        self._log(f"Snapshot restore failed: {e}")
        return False

def run_sample(self, file_path):
    if not os.path.exists(file_path):
        self._log(f"Sample not found: {file_path}")
        return False

    self._log(f"Running sample in sandbox: {file_path}")
    try:
        result = subprocess.run(["vboxmanage", "startvm", self.vm_path, "--type", "headless"], check=True)
        time.sleep(10)  # wait for VM to boot
        # Simulate drop file and run - replace with actual automation if needed
        self._log(f"Simulated drop and execution of: {file_path}")
        return True
    except Exception as e:
        self._log(f"Sample execution failed: {e}")
        return False

def collect_artifacts(self, output_dir="/tmp/sandbox_artifacts"):
    self._log(f"Collecting artifacts to: {output_dir}")
    try:
        os.makedirs(output_dir, exist_ok=True)
        # Simulated log extraction - replace with real interaction
        fake_artifact = os.path.join(output_dir, f"artifact_{self.session_id}.log")
        with open(fake_artifact, "w") as f:
            f.write("Simulated artifact content from VM run.")
        self._log("Artifacts collected.")
        return output_dir
    except Exception as e:
        self._log(f"Artifact collection failed: {e}")
        return None

def reset_vm(self):
    self._log("Resetting VM to clean snapshot.")
    return self.restore_snapshot()

if name == "main": sandbox = SandboxVM() if sandbox.restore_snapshot(): if sandbox.run_sample("/tmp/test_sample.exe"): sandbox.collect_artifacts() sandbox.reset_vm()

