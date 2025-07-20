from sentenial_core.reporting.report_generator import ReportGenerator

reporter = ReportGenerator()

# Example usage inside a detection engine
reporter.generate_threat_report({
    "source": "network_watcher",
    "severity": "CRITICAL",
    "ioc": "Suspicious DNS beaconing to c2.darkwebhost.onion",
    "timestamp": "2025-07-16T02:45:01Z"
})