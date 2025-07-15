import csv
from typing import List, Dict, Optional

class CSVSignatureManager:
    def __init__(self, filepath: str = "data/signatures/Datasigns/signature_feed.csv"):
        self.filepath = filepath
        self.signatures: List[Dict[str, str]] = []
        self.load_csv()

    def load_csv(self):
        try:
            with open(self.filepath, newline='') as csvfile:
                reader = csv.reader(csvfile)
                self.signatures = []
                for row in reader:
                    if len(row) != 3:
                        continue  # skip malformed rows
                    self.signatures.append({
                        "ip": row[0].strip(),
                        "type": row[1].strip(),
                        "severity": row[2].strip()
                    })
        except FileNotFoundError:
            print(f"[ERROR] Signature file not found: {self.filepath}")
        except Exception as e:
            print(f"[ERROR] Failed to load CSV: {e}")

    def get_all_signatures(self) -> List[Dict[str, str]]:
        return self.signatures

    def find_by_ip(self, ip: str) -> Optional[Dict[str, str]]:
        for sig in self.signatures:
            if sig["ip"] == ip:
                return sig
        return None

    def filter_by_severity(self, severity: str) -> List[Dict[str, str]]:
        return [sig for sig in self.signatures if sig["severity"].lower() == severity.lower()]

    def print_summary(self):
        print("=== Loaded Threat Signatures ===")
        for sig in self.signatures:
            print(f"{sig['ip']} -> {sig['type']} [{sig['severity']}]")

# === Example Usage ===
if __name__ == "__main__":
    manager = CSVSignatureManager()
    manager.print_summary()

    print("\n🔎 Search for IP 192.168.1.10:")
    result = manager.find_by_ip("192.168.1.10")
    print(result)

    print("\n⚠️  Filter Critical Threats:")
    for r in manager.filter_by_severity("Critical"):
        print(r)
