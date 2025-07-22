import os
import json
from executor import execute_command
from memory import remember
from ml_classifier import classify_command, train_model

# After processing commands:
train_model()
label = classify_command(cmd)
print(f"[ML] Command classified as: {label}")

COMMAND_QUEUE = "commands/agent_commands.json"

def listen_for_commands(agent_id):
    if not os.path.exists(COMMAND_QUEUE):
        return
    try:
        with open(COMMAND_QUEUE, "r") as f:
            commands = json.load(f)
        if agent_id in commands:
            for cmd in commands[agent_id]:
                print(f"[EXECUTE] {cmd}")
                execute_command(cmd)
                remember({"source": "command", "command": cmd})
            del commands[agent_id]
            with open(COMMAND_QUEUE, "w") as f:
                json.dump(commands, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Command parsing failed: {e}")
