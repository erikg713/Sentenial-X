# apps/dashboard/utils/helpers.py

"""
Dashboard Utility Helpers
-------------------------
Provides helper functions for formatting, filtering, and processing
data to be displayed in the Sentenial-X dashboard widgets.
"""

from typing import List, Dict, Any
from datetime import datetime


def format_timestamp(ts: str) -> str:
    """
    Format ISO timestamp string to a human-readable format.
    
    Args:
        ts (str): ISO-formatted timestamp.
    
    Returns:
        str: Formatted timestamp like "YYYY-MM-DD HH:MM:SS".
    """
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts


def filter_events_by_severity(events: List[Dict[str, Any]], severity: str) -> List[Dict[str, Any]]:
    """
    Filter a list of events by severity level.
    
    Args:
        events (List[Dict[str, Any]]): List of event dictionaries.
        severity (str): Severity level to filter ('info', 'medium', 'high').
    
    Returns:
        List[Dict[str, Any]]: Filtered list of events matching the severity.
    """
    return [e for e in events if e.get("severity", "").lower() == severity.lower()]


def summarize_countermeasures(events: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Summarize the number of countermeasures taken for each type of threat.
    
    Args:
        events (List[Dict[str, Any]]): List of event dictionaries with countermeasures.
    
    Returns:
        Dict[str, int]: Counts of countermeasures by type.
    """
    summary: Dict[str, int] = {}
    for event in events:
        countermeasures = event.get("countermeasures", [])
        for cm in countermeasures:
            summary[cm] = summary.get(cm, 0) + 1
    return summary


def top_attack_sources(events: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Identify top N sources with the most attacks/events.
    
    Args:
        events (List[Dict[str, Any]]): List of event dictionaries.
        top_n (int): Number of top sources to return.
    
    Returns:
        List[Dict[str, Any]]: List of sources with counts sorted descending.
    """
    source_counts: Dict[str, int] = {}
    for event in events:
        src = event.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
    
    sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)
    return [{"source": src, "count": count} for src, count in sorted_sources[:top_n]]


def safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Safely retrieve a nested key from a dictionary without raising errors.
    
    Args:
        data (Dict[str, Any]): Dictionary to retrieve value from.
        key (str): Key to lookup.
        default (Any): Default value if key is not found.
    
    Returns:
        Any: Value from dictionary or default.
    """
    return data.get(key, default)
