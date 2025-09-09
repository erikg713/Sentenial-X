import time
from heartbeat import send_heartbeat
from command_listener import listen_for_commands
from strategist import strategize
from network import start_network_server, send_message, PEERS

def run_agent():
    print(f"[BOOT] {AGENT_ID} started.")
    start_network_server(port=5000)
    while True:
        # ...
        # Example: broadcast status
        for peer in PEERS:
            send_message(peer, {"from": AGENT_ID, "status": "online"})
        time.sleep(10)
        
AGENT_ID = "sentenial-x-ai-bot"

if __name__ == "__main__":
    run_agent()
