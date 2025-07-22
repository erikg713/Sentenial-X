import time
from heartbeat import send_heartbeat
from command_listener import listen_for_commands
from strategist import strategize

AGENT_ID = "sentenial-x-ai-bot"

def run_agent():
    print(f"[BOOT] {AGENT_ID} started.")
    while True:
        send_heartbeat(AGENT_ID)
        listen_for_commands(AGENT_ID)
        strategize()
        time.sleep(10)

if __name__ == "__main__":
    run_agent()
