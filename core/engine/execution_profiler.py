"""
Profiles the execution behavior of a telemetry session.
"""

from typing import List, Dict, Any

def profile_execution_chain(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze runtime chain for behavior density and timing.
    """
    total = len(events)
    cmds = [e["command"] for e in events if "command" in e]
    exec_freq = total / (events[-1]["timestamp"] - events[0]["timestamp"]).total_seconds() if total > 1 else 0

    return {
        "total_events": total,
        "unique_commands": len(set(cmds)),
        "execution_density": round(exec_freq, 3),
        "used_commands": cmds
    }

