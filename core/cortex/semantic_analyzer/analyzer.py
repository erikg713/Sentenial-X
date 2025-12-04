#!/usr/bin/env python3
"""
Sentenial-X :: Semantic Analyzer
================================

Purpose:
    Perform semantic analysis of signals/events:
        - Extract key features
        - Map textual/log data to threat intents
        - Generate structured semantic metadata
        - Provide output consumable by ZeroDayPredictor or Orchestrator
"""

import re
from typing import Dict, Any, List
from collections import Counter

# ------------------------------------------------------------
# Core Semantic Analyzer
# ------------------------------------------------------------
class SemanticAnalyzer:
    def __init__(self, threat_keywords: Dict[str, List[str]] = None):
        """
        threat_keywords: optional dictionary mapping threat categories to keywords
        Example:
            {
                "Privilege Escalation": ["sudo", "root", "admin"],
                "Malware": ["trojan", "ransomware", "worm"]
            }
        """
        if threat_keywords is None:
            self.threat_keywords = {
                "Privilege Escalation": ["sudo", "root", "admin", "setuid"],
                "Malware Deployment": ["trojan", "worm", "virus", "payload"],
                "Data Exfiltration": ["scp", "ftp", "curl", "wget", "upload"],
                "Reconnaissance": ["scan", "nmap", "enum", "probe"],
                "Anomaly": ["anomalous", "suspicious", "unexpected"]
            }
        else:
            self.threat_keywords = threat_keywords

    # --------------------------------------------------------
    # Analyze a single event
    # --------------------------------------------------------
    def analyze(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input:
            event: dict containing:
                - log_text (str): raw event log or description
                - syscalls (list): optional list of syscalls
                - user, process, timestamp, etc.

        Output:
            semantic_info: dict with:
                - categories matched
                - keyword counts
                - anomaly flags
                - normalized features
        """
        text = event.get("log_text", "")
        syscalls = event.get("syscalls", [])
        semantic_info = {}

        # 1️⃣ Keyword matching
        category_hits = {}
        for category, keywords in self.threat_keywords.items():
            hits = sum(1 for kw in keywords if re.search(rf"\b{kw}\b", text, re.IGNORECASE))
            if hits > 0:
                category_hits[category] = hits

        semantic_info["category_hits"] = category_hits
        semantic_info["categories"] = list(category_hits.keys())
        semantic_info["keyword_count"] = sum(category_hits.values())

        # 2️⃣ Basic anomaly heuristics
        semantic_info["anomaly_flag"] = self._detect_anomaly(text, syscalls)

        # 3️⃣ Syscall features
        semantic_info["syscall_count"] = len(syscalls)
        semantic_info["unique_syscalls"] = len(set(syscalls))

        # 4️⃣ Normalized scores (0-1)
        semantic_info["normalized_keyword_score"] = min(semantic_info["keyword_count"] / 10.0, 1.0)
        semantic_info["normalized_syscall_score"] = min(len(syscalls) / 50.0, 1.0)

        return semantic_info

    # --------------------------------------------------------
    # Detect basic anomalies
    # --------------------------------------------------------
    def _detect_anomaly(self, text: str, syscalls: List[str]) -> bool:
        """
        Simple heuristics:
            - presence of suspicious keywords
            - unusually high syscall count
        """
        suspicious_keywords = ["exploit", "unauthorized", "hack", "bypass", "inject"]
        if any(re.search(rf"\b{kw}\b", text, re.IGNORECASE) for kw in suspicious_keywords):
            return True
        if len(syscalls) > 20:  # arbitrary threshold
            return True
        return False

# ------------------------------------------------------------
# Optional CLI interface
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Semantic Analyzer CLI")
    parser.add_argument("--input", required=True, help="JSON file with events")
    parser.add_argument("--output", required=False, help="Output JSON file path")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        events = json.load(f)

    analyzer = SemanticAnalyzer()
    results = [analyzer.analyze(event) for event in events]

    output_json = json.dumps(results, indent=4)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"Results saved to {args.output}")
    else:
        print(output_json)
