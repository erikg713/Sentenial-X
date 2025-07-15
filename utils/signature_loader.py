import csv

class CSVSignatureManager:
    def __init__(self, filepath):
        self.filepath = filepath
        self.signatures = []
        self.load_signatures()

    def load_signatures(self):
        try:
            with open(self.filepath, "r", newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.signatures = [row for row in reader]
        except FileNotFoundError:
            print(f"[!] Signature file not found: {self.filepath}")
        except Exception as e:
            print(f"[ERROR] Loading signatures: {e}")

    def search_by_ip(self, ip_address):
        return next((row for row in self.signatures if row["IP"] == ip_address), None)

    def filter_by_severity(self, level="Critical"):
        return [row for row in self.signatures if row["Severity"].lower() == level.lower()]

# === Example Usage ===
if __name__ == "__main__":
    manager = CSVSignatureManager("data/signatures/Datasigns/signature_feed.csv")

    print("\n🔎 Search for IP 192.168.1.10:")
    print(manager.search_by_ip("192.168.1.10"))

    print("\n⚠️  Filter Critical Threats:")
    print(manager.filter_by_severity("Critical"))
