import csv
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Union

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class LedgerEntry:
    entry_id: str
    timestamp: str  # ISO 8601 format preferred
    data: Dict[str, Any]
    signature: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "data": self.data,
            "signature": self.signature,
        }

    def hash_entry(self) -> str:
        """
        Returns a SHA-256 hash of the entry for integrity verification.
        """
        entry_json = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(entry_json.encode("utf-8")).hexdigest()

    @property
    def parsed_timestamp(self) -> datetime:
        return datetime.fromisoformat(self.timestamp)

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
            logger.error(f"Ledger file not found: {self.source}")
            raise FileNotFoundError(f"Ledger file not found: {self.source}")

        suffix = self.source.suffix.lower()
        if suffix == ".json":
            with self.source.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                logger.error("JSON ledger must be a list of entries.")
                raise ValueError("JSON ledger must be a list of entries.")
            self.entries = [LedgerEntry(**item) for item in data]
        elif suffix == ".csv":
            with self.source.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.entries = [
                    LedgerEntry(
                        entry_id=row["entry_id"],
                        timestamp=row["timestamp"],
                        data=json.loads(row["data"]),
                        signature=row.get("signature"),
                    )
                    for row in reader
                ]
        else:
            logger.error(f"Unsupported ledger file format: {suffix}")
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
            try:
                entry.parsed_timestamp
            except Exception:
                errors.append(f"Invalid timestamp format for entry_id {entry.entry_id}: {entry.timestamp}")
            seen_ids.add(entry.entry_id)
        return errors

    def sequence_entries(self) -> List[LedgerEntry]:
        """
        Returns the entries sorted by timestamp.
        """
        return sorted(self.entries, key=lambda e: e.parsed_timestamp)

    def audit_trail(self) -> Iterator[Dict[str, Any]]:
        """
        Yields each entry's hash and metadata for audit purposes.
        """
        for entry in self.sequence_entries():
            yield {
                "entry_id": entry.entry_id,
                "timestamp": entry.timestamp,
                "hash": entry.hash_entry(),
                "signature": entry.signature,
            }

    def report(self, output: Union[str, Path]) -> None:
        """
        Writes a simple audit report as JSON.
        """
        output_path = Path(output)
        report_data = list(self.audit_trail())
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        logger.info(f"Audit report written to {output_path.resolve()}")

# Configure logging for production code; consumers can override as needed.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
