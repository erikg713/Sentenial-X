import importlib
import os
import re
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load threat signatures from external JSON
SIGNATURES_FILE = os.path.join(os.path.dirname(__file__), "signatures.json")
try:
    with open(SIGNATURES_FILE, "r") as f:
        SIGNATURES = json.load(f)
except FileNotFoundError:
    SIGNATURES = {}  # Default to empty if file is not found
    logging.warning("Signature file not found. Using empty signature set.")

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
        if re.search(re.escape(keyword), payload, re.IGNORECASE)
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

    def execute_module(file):
        module_name = f"core.ids.{file[:-3]}"
        try:
            module = importlib.import_module(module_name)
            if not hasattr(module, "run_detection"):
                raise AttributeError("Module missing 'run_detection' function")
            result = module.run_detection(target)
            result["module"] = file[:-3]
            if not silent:
                print(f"[IDS] {file[:-3]}: {result.get('status', 'N/A')} - {result.get('message', '')}")
            return result
        except Exception as exc:
            error_message = {
                "module": file[:-3],
                "status": "error",
                "message": str(exc),
            }
            logging.error(f"Error in module {file[:-3]}: {exc}")
            return error_message

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(execute_module, ids_files))

    return results
