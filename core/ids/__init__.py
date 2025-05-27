# core/ids/__init__.py

import importlib
import os
from datetime import datetime

# Expandable dictionary of threat signatures
SIGNATURES = {
    "sql": [
        "' OR 1=1",
        "SELECT * FROM",
        "DROP TABLE",
    ],
    "xss": [
        "<script>",
        "onerror=",
        "alert(",
    ],
    "brute": [
        "admin",
        "password123",
        "login",
    ],
}

def detect_intrusion(payload: str) -> dict:
    """
    Scans the input payload for suspicious patterns defined in SIGNATURES.

    Args:
        payload (str): The incoming data to be checked.

    Returns:
        dict: Detection result, including timestamp and match details if threats are found.
    """
    matches = [
        (category, keyword)
        for category, keywords in SIGNATURES.items()
        for keyword in keywords
        if keyword.lower() in payload.lower()
    ]

    if matches:
        return {
            "timestamp": datetime.now().isoformat(),
            "detected": True,
            "matches": matches,
        }
    return {"detected": False}

def run_all_ids(target: str, silent: bool = False) -> list:
    """
    Dynamically runs all IDS modules in the current directory ending with '_ids.py'.

    Args:
        target (str): The data to be processed by each IDS module.
        silent (bool): If True, suppresses output to stdout.

    Returns:
        list: Results from each IDS module.
    """
    base_path = os.path.dirname(__file__)
    results = []

    ids_files = [
        f for f in os.listdir(base_path)
        if f.endswith("_ids.py") and f != os.path.basename(__file__)
    ]

    for file in ids_files:
        module_name = f"core.ids.{file[:-3]}"
        try:
            module = importlib.import_module(module_name)
            if not hasattr(module, "run_detection"):
                raise AttributeError("Module missing 'run_detection' function")
            result = module.run_detection(target)
            result["module"] = file[:-3]
            results.append(result)
            if not silent:
                print(f"[IDS] {file[:-3]}: {result.get('status', 'N/A')} - {result.get('message', '')}")
        except Exception as exc:
            results.append({
                "module": file[:-3],
                "status": "error",
                "message": str(exc)
            })
            if not silent:
                print(f"[IDS] {file[:-3]}: error - {exc}")

    return results
