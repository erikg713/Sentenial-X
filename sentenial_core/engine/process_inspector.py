import psutil
import hashlib
from sentenial_core.reporting.report_generator import ReportGenerator

reporter = ReportGenerator()

# Example usage inside a detection engine
reporter.generate_threat_report({
    "source": "network_watcher",
    "severity": "CRITICAL",
    "ioc": "Suspicious DNS beaconing to c2.darkwebhost.onion",
    "timestamp": "2025-07-16T02:45:01Z"
})

class ProcessInspector:
    def __init__(self):
        self.known_hashes = set()

    def _hash_exe(self, exe_path):
        try:
            with open(exe_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return None

    def scan(self):
        findings = []
        for proc in psutil.process_iter(['pid', 'exe', 'name']):
            try:
                if proc.info["exe"]:
                    h = self._hash_exe(proc.info["exe"])
                    if h and h not in self.known_hashes:
                        self.known_hashes.add(h)
                        findings.append(f"New executable detected: {proc.info['name']} (PID: {proc.info['pid']})")
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        return findings