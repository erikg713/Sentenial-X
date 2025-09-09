import json
from memory import MEMORY_FILE

def learn_from_memory():
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE, "r") as f:
        memory = json.load(f)
    stats = {}
    for entry in memory:
        cmd = entry["event"].get("command")
        if cmd:
            stats[cmd] = stats.get(cmd, 0) + 1
    return stats
