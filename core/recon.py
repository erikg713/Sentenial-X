# core/recon.py

import socket
import requests
import subprocess
import json
import re
from datetime import datetime

def get_ip(domain):
    try:
        return socket.gethostbyname(domain)
    except Exception:
        return None

def fetch_headers(url):
    try:
        response = requests.get(url, timeout=5)
        return dict(response.headers)
    except Exception:
        return {}

def detect_tech_stack(url):
    headers = fetch_headers(url)
    stack = []
    if "x-powered-by" in headers:
        stack.append(headers["x-powered-by"])
    if "server" in headers:
        stack.append(headers["server"])
    return stack

def whois_lookup(domain):
    try:
        result = subprocess.check_output(["whois", domain], text=True)
        return result
    except Exception:
        return "WHOIS lookup failed"

def extract_emails(text):
    return re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)

def perform_full_recon(domain):
    ip = get_ip(domain)
    headers = fetch_headers(f"http://{domain}")
    tech = detect_tech_stack(f"http://{domain}")
    whois_data = whois_lookup(domain)
    emails = extract_emails(whois_data)

    return {
        "domain": domain,
        "ip": ip,
        "headers": headers,
        "tech_stack": tech,
        "emails": emails,
        "whois": whois_data,
        "timestamp": datetime.now().isoformat()
    }
