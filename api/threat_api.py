# Example data loader from logs/db
def get_threat_data():
    return [
        {
            "id": 1,
            "severity": "Critical",
            "name": "SQL Injection",
            "description": "Injection attempt detected on login form.",
            "tags": ["sql", "web", "critical"]
        },
        {
            "id": 2,
            "severity": "Medium",
            "name": "Port Scan",
            "description": "Suspicious port scanning behavior.",
            "tags": ["scan", "network"]
        },
        # Add actual data from your threat engine here
    ]