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
# core/ids/__init__.py

import importlib
import os

def run_all_ids(target, silent=False):
    base_path = os.path.dirname(__file__)
    results = []

    for file in os.listdir(base_path):
        if file.endswith("_ids.py"):
            module_name = file[:-3]
            module = importlib.import_module(f"core.ids.{module_name}")
            try:
                result = module.run_detection(target)
                result["module"] = module_name
                results.append(result)
                if not silent:
                    print(f"[IDS] {module_name}: {result['status']} - {result['message']}")
            except Exception as e:
                results.append({
                    "module": module_name,
                    "status": "error",
                    "message": str(e)
                })
    
    return results
