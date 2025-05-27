def build_timeline(events: list) -> list:
    """
    Sorts events by timestamp for forensic analysis.
    Each event should be a dict with at least a 'timestamp' key.
    """
    return sorted(events, key=lambda e: e['timestamp'])