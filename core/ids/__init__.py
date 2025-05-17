# core/ids/__init__.py

from datetime import datetime

# Dummy threat signatures (expandable)
SIGNATURES = {
    "sql": ["' OR 1=1", "SELECT * FROM", "DROP TABLE"],
    "xss": ["<script>", "onerror=", "alert("],
    "brute": ["admin", "password123", "login"]
}

def detect_intrusion(payload):
    matches = []
    for category, keywords in SIGNATURES.items():
        for k in keywords:
            if k.lower() in payload.lower():
                matches.append((category, k))
    if matches:
        return {
            "timestamp": datetime.now().isoformat(),
            "detected": True,
            "matches": matches
        }
    return {"detected": False}
