"""
Correlates telemetry into coherent chains based on session, time, and parentage.
"""

from typing import List, Dict, Any
from collections import defaultdict
import datetime

MAX_TIME_DELTA_SEC = 30

def correlate_events(events: List[Dict[str, Any]]) -> List[List[Dict]]:
    """
    Group events into correlated chains by session_id and temporal proximity.
    """
    chains = defaultdict(list)
    for event in sorted(events, key=lambda x: x.get("timestamp", "")):
        session_id = event.get("session_id", "unknown")
        chains[session_id].append(event)

    correlated = []
    for chain in chains.values():
        group = []
        last_time = None
        for evt in chain:
            ts = evt.get("timestamp")
            if isinstance(ts, str):
                ts = datetime.datetime.fromisoformat(ts)
            if last_time and (ts - last_time).total_seconds() > MAX_TIME_DELTA_SEC:
                correlated.append(group)
                group = []
            group.append(evt)
            last_time = ts
        if group:
            correlated.append(group)

    return correlated

