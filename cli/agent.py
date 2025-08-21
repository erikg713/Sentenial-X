#!/usr/bin/env python3
import argparse
import subprocess
import json
import sys
from datetime import datetime

def run_command(command, args):
    """
    Executes a Sentenial-X CLI command and captures output.
    """
    full_cmd = ["./sentenial_cli_full.py", command] + args
    try:
        result = subprocess.run(full_cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

def report_result(agent_id, command, output):
    """
    Logs or reports result to memory/server.
    """
    payload = {
        "agent_id": agent_id,
        "timestamp": datetime.utcnow().isoformat(),
        "command": command,
        "output": output
    }
    print(json.dumps(payload, indent=2))  # Replace with HTTP POST if reporting centrally

def main():
    parser = argparse.ArgumentParser(description="Sentenial-X Agent CLI")
    parser.add_argument("agent_id", type=str, help="Unique Agent ID")
    parser.add_argument("command", type=str, help="Command to run (e.g., wormgpt-detector)")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Additional arguments for the command")

    args = parser.parse_args()
    output = run_command(args.command, args.args)
    report_result(args.agent_id, args.command, output)

if __name__ == "__main__":
    main()
