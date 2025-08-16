"""
Sentenial-X Reporting - report_builder.py
-----------------------------------------
Generates structured reports from threat logs, telemetry, and forensic data.
Supports multiple export formats (JSON, Markdown, PDF-ready text).
"""

import json
import datetime
from typing import Dict, Any, Optional


class ReportBuilder:
    def __init__(self, title: str = "Sentenial-X Threat Report", author: str = "Sentenial-X Engine"):
        self.title = title
        self.author = author
        self.timestamp = datetime.datetime.utcnow().isoformat()
        self.sections = []

    def add_section(self, heading: str, content: str, severity: Optional[str] = None) -> None:
        """Add a structured section to the report."""
        entry = {
            "heading": heading,
            "content": content,
            "severity": severity or "info",
            "created_at": datetime.datetime.utcnow().isoformat(),
        }
        self.sections.append(entry)

    def to_dict(self) -> Dict[str, Any]:
        """Return report as a structured Python dict."""
        return {
            "title": self.title,
            "author": self.author,
            "timestamp": self.timestamp,
            "sections": self.sections,
        }

    def to_json(self, indent: int = 2) -> str:
        """Return report as JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Return report formatted as Markdown."""
        md = [f"# {self.title}", f"**Author:** {self.author}", f"**Generated:** {self.timestamp}", ""]
        for sec in self.sections:
            sev = sec["severity"].upper()
            md.append(f"## {sec['heading']} [{sev}]")
            md.append(sec["content"])
            md.append("")
        return "\n".join(md)

    def to_text(self) -> str:
        """Return report as plain text."""
        text = [f"{self.title}", f"Author: {self.author}", f"Generated: {self.timestamp}", "-" * 40]
        for sec in self.sections:
            sev = sec["severity"].upper()
            text.append(f"[{sev}] {sec['heading']}")
            text.append(sec["content"])
            text.append("-" * 40)
        return "\n".join(text)


# Example usage
if __name__ == "__main__":
    rb = ReportBuilder(title="Forensic Analysis Report")
    rb.add_section("System Integrity", "No unauthorized modifications detected.", severity="low")
    rb.add_section("Network Activity", "Suspicious connection attempt to 185.34.12.9", severity="high")
    rb.add_section("Malware Scan", "Potential trojan detected in memory segment 0x3f2c", severity="critical")

    print("\n=== JSON ===\n")
    print(rb.to_json())

    print("\n=== Markdown ===\n")
    print(rb.to_markdown())

    print("\n=== Plain Text ===\n")
    print(rb.to_text())
