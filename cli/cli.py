import asyncio
import json
import os
import typer
import requests
import logging

from memory import enqueue_command
from config import COMMAND_POLL_INTERVAL, AGENT_ID, NETWORK_PORT

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create Typer app
app = typer.Typer(help="📡 Sentenial-X AI Agent CLI Interface")


@app.command()
def start():
    """
    🚀 Start the AI Agent in the foreground.
    """
    typer.echo("🚀 Launching Sentenial-X agent...")
    os.execvp("python3", ["python3", "agent.py"])


@app.command()
def send(command: str):
    """
    📤 Enqueue a command to the AI agent.
    """
    try:
        asyncio.run(enqueue_command(AGENT_ID, command.strip()))
        typer.echo(f"✅ Command queued: {command}")
    except Exception as e:
        logging.exception("❌ Failed to enqueue command")
        typer.echo(f"❌ Error queuing command: {e}")


@app.command()
def status():
    """
    🧠 Check the health status of the AI agent.
    """
    url = f"http://localhost:{NETWORK_PORT}/health"
    try:
        response = requests.get(url, timeout=3)
        response.raise_for_status()
        typer.echo(json.dumps(response.json(), indent=2))
    except requests.ConnectionError:
        typer.echo("❌ Agent is not running or unreachable.")
    except requests.Timeout:
        typer.echo("⏱️ Request to agent timed out.")
    except Exception as e:
        logging.exception("❌ Unexpected error during status check")
        typer.echo(f"❌ Unexpected error: {e}")


@app.command()
def broadcast(message: str):
    """
    📡 Manually broadcast a status message to peer agents.
    """
    url = f"http://localhost:{NETWORK_PORT}/message"
    payload = {"from": AGENT_ID, "msg": message}

    try:
        response = requests.post(url, json=payload, timeout=3)
        if response.status_code == 204:
            typer.echo("📢 Broadcast sent successfully.")
        else:
            typer.echo(f"⚠️ Unexpected response: {response.status_code}")
    except requests.RequestException as e:
        logging.exception("❌ Broadcast request failed")
        typer.echo(f"❌ Broadcast error: {e}")


@app.command()
def stop():
    """
    🛑 Stop the AI Agent gracefully.
    (Note: requires agent to support signal handling or PID tracking.)
    """
    try:
        os.system("pkill -f agent.py")
        typer.echo("🛑 Agent stopped.")
    except Exception as e:
        logging.exception("❌ Failed to stop agent")
        typer.echo(f"❌ Error stopping agent: {e}")
 def main():
+    banner = """
+   🚨 Sentenial-X A.I. – The Ultimate Cyber Guardian 🚨
+   Crafted for resilience. Engineered for vengeance.
+   A digital sentinel with the mind of a warrior and the reflexes of a machine.
+    """
+    print(banner)
     # existing CLI entrypoints…

import argparse
from sentenialx.ai_core import (
    detect_prompt_threat,
    log_threat_event,
    update_model,
    start_ipc_server
)
from sentenialx.ai_core.datastore import get_recent_threats


def run_scan(text):
    confidence = detect_prompt_threat(text)
    print(f"[SCAN RESULT] Confidence of threat: {confidence:.2f}")
    if confidence > 0.85:
        print("[⚠️] THREAT detected!")
        log_threat_event("ai_prompt_threat", "cli", text, confidence)
    else:
        print("[✅] No threat detected.")

def add_feedback(text, label):
    update_model(text, label)
    print(f"[🧠] Feedback recorded: '{label}' -> {text}")

def view_threats(limit):
    threats = get_recent_threats(limit)
    print(f"\n[🧾] Showing last {len(threats)} threat(s):")
    for row in threats:
        print(f"{row[1]} | {row[2]} | {row[4]} | Confidence: {row[5]}")

def main():
    parser = argparse.ArgumentParser(
        description="🧠 Sentenial X A.I. CLI — Threat Scanner & Daemon Interface"
    )
    subparsers = parser.add_subparsers(dest="command")

# 🔁 watch live threats
    subparsers.add_parser("watch", help="Stream live threats from log DB")

    # 📄 scan a text file line-by-line
    file_parser = subparsers.add_parser("scanfile", help="Scan lines from a file")
    file_parser.add_argument("file_path", type=str, help="Path to file")

    # 💣 simulate payload
    sim_parser = subparsers.add_parser("simulate", help="Run a ransomware test payload")

    # 🔐 defend mode
    subparsers.add_parser("defend", help="Auto-monitor stdin for threats (stealth mode)")

    # ⛔ shutdown daemon
    subparsers.add_parser("shutdown", help="Trigger shutdown across agents")

    # 🔍 scan
    scan_parser = subparsers.add_parser("scan", help="Scan a string for threats")
    scan_parser.add_argument("text", type=str, help="Text to analyze")

    # 🧠 feedback
    feedback_parser = subparsers.add_parser("feedback", help="Provide feedback to model")
    feedback_parser.add_argument("text", type=str, help="Text input")
    feedback_parser.add_argument("label", type=str, choices=["malicious", "safe"], help="Label")

    # 📜 view logs
    logs_parser = subparsers.add_parser("logs", help="View recent threat events")
    logs_parser.add_argument("--limit", type=int, default=10, help="How many entries to show")

    # 🚀 daemon
    daemon_parser = subparsers.add_parser("daemon", help="Run the AI daemon server")

    args = parser.parse_args()

    if args.command == "scan":
        run_scan(args.text)
    elif args.command == "feedback":
        add_feedback(args.text, args.label)
    elif args.command == "logs":
        view_threats(args.limit)
    elif args.command == "daemon":
        print("[🚀] Launching Sentenial X Daemon (ZeroMQ IPC on port 5556)...")
        start_ipc_server()
        input("[Press Enter to stop the daemon...]\n")
    else:
        parser.print_help()
        
if __name__ == "__main__":
    app()
