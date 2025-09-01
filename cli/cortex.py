"""
cli/cortex.py

NLP Threat Analysis Module for Sentenial-X.

Provides async functions to analyze log files or event streams
using built-in NLP models and return structured threat findings.
"""

import asyncio
import logging
import re
from datetime import datetime

from cli import memory

logger = logging.getLogger("CortexModule")


# ------------------------------
# Helper functions
# ------------------------------
def extract_entities(line: str) -> dict:
    """
    Simple NLP-like entity extraction from a log line.
    In production, replace with advanced NLP model.
    """
    entities = {}
    # Example: IP detection
    ip_match = re.findall(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", line)
    if ip_match:
        entities["ips"] = ip_match

    # Example: user detection (basic pattern)
    user_match = re.findall(r"user=([\w\-]+)", line)
    if user_match:
        entities["users"] = user_match

    return entities


def classify_line(line: str, filter_expr: str = None) -> dict:
    """
    Classify a line for suspicious patterns based on a simple keyword filter.
    """
    result = {"line": line, "risk": "low", "entities": {}}

    if filter_expr and re.search(filter_expr, line, re.IGNORECASE):
        result["risk"] = "medium"

    # Example pattern matching for high-risk terms
    high_risk_terms = ["failed login", "unauthorized", "error", "segfault", "exploit"]
    for term in high_risk_terms:
        if term in line.lower():
            result["risk"] = "high"
            break

    # Extract entities
    result["entities"] = extract_entities(line)
    return result


# ------------------------------
# Core async analyzer
# ------------------------------
async def analyze(source: str, filter_expr: str = None) -> dict:
    """
    Analyze log file or event source asynchronously.

    Args:
        source (str): Path to log file or event source
        filter_expr (str, optional): Regex filter for targeted analysis

    Returns:
        dict: structured findings
    """
    findings = []

    try:
        logger.info(f"Starting analysis on source: {source}")
        async with aiofiles.open(source, mode="r") as f:
            async for line in f:
                line = line.strip()
                if not line:
                    continue
                result = classify_line(line, filter_expr)
                if result["risk"] != "low":
                    findings.append(result)

        output = {
            "action": "cortex",
            "source": source,
            "findings": findings,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Log to memory
        await memory.enqueue_command(
            action="cortex",
            params={"source": source, "filter": filter_expr},
            result=output,
        )

        logger.info(f"Analysis complete: {len(findings)} suspicious lines found")
        return output

    except FileNotFoundError:
        logger.error(f"Source file not found: {source}")
        return {"action": "cortex", "error": "source_not_found", "source": source}
    except Exception as e:
        logger.exception(f"Unexpected error during cortex analysis: {e}")
        return {"action": "cortex", "error": str(e), "source": source}


# ------------------------------
# Optional standalone test
# ------------------------------
if __name__ == "__main__":
    import aiofiles
    import argparse

    parser = argparse.ArgumentParser(description="Cortex NLP Threat Analyzer")
    parser.add_argument("-s", "--source", required=True, help="Log file path")
    parser.add_argument("-f", "--filter", help="Optional regex filter")
    args = parser.parse_args()

    result = asyncio.run(analyze(args.source, args.filter))
    print(result)
