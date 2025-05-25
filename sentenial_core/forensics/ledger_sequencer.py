import csv
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Union

class LedgerEntry:
    """
    Represents a single ledger entry.
    """
    def __init__(self, entry_id: str, timestamp: str, data: Dict[str, Any], signature: Optional[str] = None):
        self.entry_id = entry_id
        self.timestamp = timestamp
        self.data = data
        self.signature = signature

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "data": self.data,
            "signature": self.signature
        }

    def hash_entry(self) -> str:
        """
        Returns a SHA-256 hash of the entry for integrity verification.
        """
        entry_json = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(entry_json.encode('utf-8')).hexdigest()

class LedgerSequencer:
    """
    Handles sequencing, validation, and auditing of ledger entries.
    """

    def __init__(self, source: Union[str, Path]):
        self.source = Path(source)
        self.entries: List[LedgerEntry] = []

    def load_entries(self) -> None:
        """
        Loads ledger entries from a JSON or CSV file.
        """
        if not self.source.exists():
            raise FileNotFoundError(f"Ledger file not found: {self.source}")

        if self.source.suffix.lower() == ".json":
            with self.source.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self.entries = [LedgerEntry(**item) for item in data]
        elif self.source.suffix.lower() == ".csv":
            with self.source.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.entries = [
                    LedgerEntry(
                        entry_id=row["entry_id"],
                        timestamp=row["timestamp"],
                        data=json.loads(row["data"]),
                        signature=row.get("signature")
                    )
                    for row in reader
                ]
        else:
            raise ValueError("Unsupported ledger file format. Use JSON or CSV.")

    def validate_entries(self) -> List[str]:
        """
        Validates entries and returns a list of error messages.
        """
        errors = []
        seen_ids = set()
        for entry in self.entries:
            if not entry.entry_id:
                errors.append(f"Missing entry_id at timestamp {entry.timestamp}")
            if entry.entry_id in seen_ids:
                errors.append(f"Duplicate entry_id: {entry.entry_id}")
            seen_ids.add(entry.entry_id)
            # Add more validation logic as needed
        return errors

    def sequence_entries(self) -> List[LedgerEntry]:
        """
        Returns the entries sorted by timestamp.
        """
        return sorted(self.entries, key=lambda e: e.timestamp)

    def audit_trail(self) -> Iterator[Dict[str, Any]]:
        """
        Yields each entry's hash and metadata for audit purposes.
        """
        for entry in self.sequence_entries():
            yield {
                "entry_id": entry.entry_id,
                "timestamp": entry.timestamp,
                "hash": entry.hash_entry(),
                "signature": entry.signature
            }

    def report(self, output: Union[str, Path]) -> None:
        """
        Writes a simple audit report as JSON.
        """
        output_path = Path(output)
        report_data = list(self.audit_trail())
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"Audit report written to {output_path.resolve()}")

# Example usage (to be removed/commented out in production)
# if __name__ == "__main__":
#     sequencer = LedgerSequencer("ledger.json")
#     sequencer.load_entries()
#     errors = sequencer.validate_entries()
#     if errors:
#         print("Validation errors:", errors)
#     else:
#         sequencer.report("audit_report.json")
