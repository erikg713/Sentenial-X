import os
import json
from typing import List, Dict, Optional

class DatasignsManager:
    def __init__(self, base_path="data/signatures/Datasigns"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)
        self.signatures = {}  # key: signature id, value: data dict
        self.load_signatures()

    def load_signatures(self):
        """Load all signature files (JSON) from the base_path."""
        self.signatures.clear()
        for filename in os.listdir(self.base_path):
            if filename.endswith(".json"):
                full_path = os.path.join(self.base_path, filename)
                try:
                    with open(full_path, "r") as f:
                        data = json.load(f)
                        sig_id = data.get("id") or filename.replace(".json", "")
                        self.signatures[sig_id] = data
                except Exception as e:
                    print(f"[WARN] Failed to load {filename}: {e}")

    def get_signature(self, sig_id: str) -> Optional[Dict]:
        """Return signature data by ID."""
        return self.signatures.get(sig_id)

    def search_signatures(self, keyword: str) -> List[Dict]:
        """Search signatures by keyword in name or description."""
        results = []
        for sig in self.signatures.values():
            name = sig.get("name", "").lower()
            desc = sig.get("description", "").lower()
            if keyword.lower() in name or keyword.lower() in desc:
                results.append(sig)
        return results

    def add_signature(self, sig_data: Dict):
        """Add a new signature and save it as JSON file."""
        sig_id = sig_data.get("id")
        if not sig_id:
            raise ValueError("Signature data must have an 'id' field")
        self.signatures[sig_id] = sig_data

        filepath = os.path.join(self.base_path, f"{sig_id}.json")
        with open(filepath, "w") as f:
            json.dump(sig_data, f, indent=2)

    def remove_signature(self, sig_id: str) -> bool:
        """Remove a signature by ID. Returns True if removed."""
        if sig_id in self.signatures:
            del self.signatures[sig_id]
            filepath = os.path.join(self.base_path, f"{sig_id}.json")
            if os.path.exists(filepath):
                os.remove(filepath)
            return True
        return False

    def list_all_signatures(self) -> List[str]:
        """Return a list of all signature IDs."""
        return list(self.signatures.keys())

if __name__ == "__main__":
    dm = DatasignsManager()

    print("Loaded signatures:", dm.list_all_signatures())

    # Example: search for 'ransomware'
    results = dm.search_signatures("ransomware")
    print(f"Search results for 'ransomware': {results}")

    # Example: add a signature
    new_sig = {
        "id": "sig_1234",
        "name": "Example Malware Signature",
        "description": "Detects example malware behavior",
        "pattern": "evil_function_call",
        "tags": ["malware", "example"],
    }
    dm.add_signature(new_sig)
    print("Added new signature 'sig_1234'")

    # Reload and check
    dm.load_signatures()
    print("Signatures after reload:", dm.list_all_signatures())

