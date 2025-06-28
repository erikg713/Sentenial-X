# core/recon.py

import socket
import requests
import subprocess
import re
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime

# Configure logging for debugging and audit trails
logging.basicConfig(
    filename='recon.log',
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

DEFAULT_TIMEOUT = 8  # seconds
HEADERS = {"User-Agent": "Sentenial-X Recon/1.0"}

def get_ip(domain: str) -> Optional[str]:
    """
    Resolve the IP address of a domain.
    """
    try:
        ip = socket.gethostbyname(domain)
        logging.info(f"Resolved IP for {domain}: {ip}")
        return ip
    except Exception as ex:
        logging.warning(f"IP resolution failed for {domain}: {ex}")
        return None

def fetch_headers(url: str) -> Dict[str, str]:
    """
    Fetch HTTP response headers for a given URL.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=DEFAULT_TIMEOUT)
        logging.info(f"Fetched headers for {url}")
        return dict(response.headers)
    except Exception as ex:
        logging.warning(f"Header fetch failed for {url}: {ex}")
        return {}

def detect_tech_stack(url: str) -> List[str]:
    """
    Infer technology stack from HTTP headers.
    """
    headers = fetch_headers(url)
    stack = []
    for key in ("x-powered-by", "server"):
        value = headers.get(key)
        if value:
            stack.append(value)
    logging.info(f"Detected tech stack for {url}: {stack}")
    return stack

def whois_lookup(domain: str) -> str:
    """
    Run a WHOIS lookup for a domain.
    """
    try:
        result = subprocess.check_output(
            ["whois", domain],
            text=True, timeout=15
        )
        logging.info(f"WHOIS lookup successful for {domain}")
        return result
    except subprocess.TimeoutExpired:
        logging.error(f"WHOIS lookup timed out for {domain}")
        return "WHOIS lookup timed out"
    except Exception as ex:
        logging.error(f"WHOIS lookup failed for {domain}: {ex}")
        return "WHOIS lookup failed"

def extract_emails(text: str) -> List[str]:
    """
    Extract email addresses from the provided text.
    """
    pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    emails = re.findall(pattern, text)
    unique_emails = sorted(set(emails))
    logging.info(f"Extracted emails: {unique_emails}")
    return unique_emails

def perform_full_recon(domain: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Perform a comprehensive reconnaissance on a domain.
    """
    protocol = 'https'  # Try HTTPS first
    url = f"{protocol}://{domain}"
    ip = get_ip(domain)
    headers = fetch_headers(url)
    if not headers:  # Fallback to HTTP if HTTPS fails
        protocol = 'http'
        url = f"{protocol}://{domain}"
        headers = fetch_headers(url)
    tech_stack = detect_tech_stack(url)
    whois_data = whois_lookup(domain)
    emails = extract_emails(whois_data)
    result = {
        "domain": domain,
        "ip": ip,
        "headers": headers,
        "tech_stack": tech_stack,
        "emails": emails,
        "whois": whois_data,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    if verbose:
        print(json.dumps(result, indent=2))
    logging.info(f"Recon complete for {domain}")
    return result

if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python recon.py <domain>")
        sys.exit(1)
    domain = sys.argv[1]
    output = perform_full_recon(domain, verbose=True)
