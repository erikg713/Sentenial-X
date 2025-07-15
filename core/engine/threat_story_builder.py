"""
Builds narrative threat stories from correlated chains.
"""

from typing import List, Dict, Any
from core.engine.stage_classifier import classify_stage

def build_story(chain: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a threat timeline with stage classification.
    """
    timeline = []
    for evt in chain:
        stage = classify_stage(evt)
        timeline.append({
            "timestamp": evt.get("timestamp"),
            "command": evt.get("command"),
            "stage": stage
        })

    return {
        "session_id": chain[0].get("session_id", "unknown"),
        "steps": timeline,
        "unique_stages": list({step["stage"] for step in timeline if step["stage"] != "Unknown"})
    }

