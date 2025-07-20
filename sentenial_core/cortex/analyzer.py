import time
import logging
import threading
from datetime import datetime
from sentenial_core.reporting.report_generator import ReportGenerator

logger = logging.getLogger("ThreatAnalyzerService")
logging.basicConfig(level=logging.INFO)

class ThreatAnalyzer:
    def __init__(self):
        self.reporter = ReportGenerator()

    def report_threat(self, source: str, severity: str, ioc: str, timestamp: str = None):
        if severity not in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            logger.error(f"Invalid severity level: {severity}")
            return

        report_data = {
            "source": source,
            "severity": severity,
            "ioc": ioc,
            "timestamp": timestamp or datetime.utcnow().isoformat() + "Z"
        }

        try:
            self.reporter.generate_threat_report(report_data)
            logger.info(f"Threat reported: {report_data}")
        except Exception as e:
            logger.exception("Failed to generate threat report.")

class ThreatMonitorDaemon:
    def __init__(self, interval=10):
        self.analyzer = ThreatAnalyzer()
        self.interval = interval  # Time between scans
        self.running = False

    def _mock_detect_threat(self):
        # Replace with actual threat detection engine logic
        return {
            "source": "network_watcher",
            "severity": "CRITICAL",
            "ioc": "Suspicious DNS beaconing to c2.darkwebhost.onion"
        }

    def monitor_loop(self):
        logger.info("Threat monitor started.")
        while self.running:
            try:
                threat = self._mock_detect_threat()
                self.analyzer.report_threat(
                    source=threat["source"],
                    severity=threat["severity"],
                    ioc=threat["ioc"]
                )
                time.sleep(self.interval)
            except Exception as e:
                logger.exception("Error in monitoring loop.")

    def start(self):
        self.running = True
        threading.Thread(target=self.monitor_loop, daemon=True).start()

    def stop(self):
        self.running = False
        logger.info("Threat monitor stopped.")

# Optional: Run as standalone script
if __name__ == "__main__":
    daemon = ThreatMonitorDaemon(interval=15)
    try:
        daemon.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        daemon.stop()

from sentenial_core.reporting.report_generator import ReportGenerator

reporter = ReportGenerator()

# Example usage inside a detection engine
reporter.generate_threat_report({
    "source": "network_watcher",
    "severity": "CRITICAL",
    "ioc": "Suspicious DNS beaconing to c2.darkwebhost.onion",
    "timestamp": "2025-07-16T02:45:01Z"
})