# threat-engine/app/sample_data.py
SAMPLE_ALERTS = [
    {
        "id": "alert-001",
        "severity": "critical",
        "title": "Fileless persistence attempt detected",
        "description": "Suspicious PowerShell activity exhibiting living-off-the-land technique.",
    },
    {
        "id": "alert-002",
        "severity": "high",
        "title": "C2 beaconing pattern",
        "description": "Periodic HTTP beacons to suspicious domain",
    },
    {
        "id": "alert-003",
        "severity": "medium",
        "title": "Brute force attempt",
        "description": "Multiple failed logins from same IP",
    },
]