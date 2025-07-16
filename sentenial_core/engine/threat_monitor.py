import os
import time
import logging
from engine.file_integrity import FileIntegrityMonitor
from engine.process_inspector import ProcessInspector
from engine.network_watcher import NetworkWatcher
from engine.alert_dispatcher import AlertDispatcher
from engine.incident_logger import IncidentLogger

class ThreatMonitor:
    def __init__(self):
        self.logger = logging.getLogger("ThreatMonitor")
        self.alerts = AlertDispatcher()
        self.incident_db = IncidentLogger("threat_incidents.db")

        self.fs_monitor = FileIntegrityMonitor(paths=["/home", "/etc", "/var/www"])
        self.proc_inspector = ProcessInspector()
        self.net_monitor = NetworkWatcher()

    def run(self):
        self.logger.info("Starting Threat Monitor")
        try:
            while True:
                self.check_filesystem()
                self.check_processes()
                self.check_network()
                time.sleep(5)
        except KeyboardInterrupt:
            self.logger.info("Stopping Threat Monitor.")

    def check_filesystem(self):
        anomalies = self.fs_monitor.scan()
        for issue in anomalies:
            self.logger.warning(f"[FS] {issue}")
            self.alerts.send(f"Filesystem Alert: {issue}")
            self.incident_db.log("filesystem", issue)

    def check_processes(self):
        threats = self.proc_inspector.scan()
        for threat in threats:
            self.logger.warning(f"[Process] {threat}")
            self.alerts.send(f"Process Threat: {threat}")
            self.incident_db.log("process", threat)

    def check_network(self):
        events = self.net_monitor.scan()
        for event in events:
            self.logger.warning(f"[Network] {event}")
            self.alerts.send(f"Network Threat: {event}")
            self.incident_db.log("network", event)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    monitor = ThreatMonitor()
    monitor.run()