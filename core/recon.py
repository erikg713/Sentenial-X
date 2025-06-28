# core/recon.py

import socket
import requests
import subprocess
import re
import logging
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime
import json
import argparse
import sys
import time

# --- Configuration ---
USER_AGENT = "Sentenial-X-Recon/3.0 (+https://github.com/erikg713/Sentenial-X-A.I.)"
DEFAULT_TIMEOUT = 8
DEFAULT_HEADERS = {"User-Agent": USER_AGENT}
COMMON_PORTS = [21, 22, 23, 25, 53, 80, 110, 143, 443, 445, 3389]  # Extend as needed

# --- Logging ---
logging.basicConfig(
    filename='recon.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s]: %(message)s'
)

# --- Utilities ---
def retry(func, *args, retries=2, delay=1, **kwargs):
    """Retry a function on exception."""
    for attempt in range(retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as ex:
            if attempt == retries:
                logging.error(f"{func.__name__} failed after {retries + 1} attempts: {ex}")
                raise
            logging.warning(f"{func.__name__} failed (attempt {attempt+1}), retrying: {ex}")
            time.sleep(delay)

# --- Recon Modules ---
def get_ip(domain: str) -> Optional[str]:
    """Resolve the IP address of a domain."""
    try:
        ip = socket.gethostbyname(domain)
        logging.info(f"Resolved IP for {domain}: {ip}")
        return ip
    except Exception as e:
        logging.error(f"IP resolution failed for {domain}: {e}")
        return None

def fetch_headers(url: str, headers=None, proxies=None) -> Dict[str, str]:
    """
    Fetch HTTP(S) response headers for a given URL.
    Returns an empty dict on failure.
    """
    _headers = DEFAULT_HEADERS.copy()
    if headers:
        _headers.update(headers)
    try:
        resp = requests.get(url, headers=_headers, timeout=DEFAULT_TIMEOUT, proxies=proxies)
        logging.info(f"Fetched headers for {url} ({resp.status_code})")
        return dict(resp.headers)
    except Exception as e:
        logging.warning(f"Header fetch failed for {url}: {e}")
        return {}

def detect_tech_stack(headers: Dict[str, str]) -> List[str]:
    """Infer technology stack from HTTP(S) response headers."""
    stack = []
    for key in ("x-powered-by", "server"):
        if key in headers:
            stack.append(headers[key])
    logging.info(f"Detected tech stack: {stack}")
    return stack

def whois_lookup(domain: str) -> str:
    """Perform a WHOIS lookup for the given domain."""
    try:
        result = subprocess.check_output(["whois", domain], text=True, timeout=15)
        logging.info(f"WHOIS lookup successful for {domain}")
        return result
    except subprocess.TimeoutExpired:
        logging.error(f"WHOIS lookup timed out for {domain}")
        return "WHOIS lookup timed out"
    except Exception as e:
        logging.error(f"WHOIS lookup failed for {domain}: {e}")
        return "WHOIS lookup failed"

def extract_emails(text: str) -> List[str]:
    """Extract unique email addresses from text."""
    pattern = r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b"
    emails = re.findall(pattern, text)
    emails = sorted(set(emails))
    logging.info(f"Extracted emails: {emails}")
    return emails

def scan_ports(ip: str, ports: List[int]=COMMON_PORTS, timeout: float=1.0) -> Dict[int, str]:
    """
    Scan the target IP for open common ports.
    Returns a dict of {port: 'open'/'closed'}.
    """
    results = {}
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            try:
                result = sock.connect_ex((ip, port))
                if result == 0:
                    results[port] = 'open'
                    logging.info(f"Port {port} open on {ip}")
                else:
                    results[port] = 'closed'
            except Exception as e:
                results[port] = 'error'
                logging.warning(f"Port scan error on {ip}:{port} - {e}")
    return results

# --- Orchestrator ---
def perform_full_recon(
    domain: str,
    verbose: bool = False,
    proxies: Optional[Dict[str, str]] = None,
    headers: Optional[Dict[str, str]] = None,
    scan: bool = True,
) -> Dict[str, Any]:
    """
    Perform comprehensive recon on the given domain.
    Attempts HTTPS first, falls back to HTTP if necessary.
    """
    def try_url(proto: str) -> str:
        return f"{proto}://{domain}"

    ip = retry(get_ip, domain)
    url = try_url("https")
    headers_resp = fetch_headers(url, headers, proxies)
    if not headers_resp:
        url = try_url("http")
        headers_resp = fetch_headers(url, headers, proxies)
    tech_stack = detect_tech_stack(headers_resp)
    whois_data = retry(whois_lookup, domain)
    emails = extract_emails(whois_data)
    ports = scan_ports(ip) if scan and ip else {}

    result = {
        "domain": domain,
        "ip": ip,
        "headers": headers_resp,
        "tech_stack": tech_stack,
        "emails": emails,
        "whois": whois_data,
        "ports": ports,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    if verbose:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    logging.info(f"Recon complete for {domain}")
    return result

# --- CLI Interface ---
def main():
    parser = argparse.ArgumentParser(
        description="Sentenial-X Recon: Professional domain intelligence gathering."
    )
    parser.add_argument("domain", help="Target domain for reconnaissance")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print full JSON output to console")
    parser.add_argument("--proxy", help="HTTP/HTTPS proxy URL (e.g., http://localhost:8080)")
    parser.add_argument("--no-scan", action="store_true", help="Skip port scanning")
    parser.add_argument("--header", action="append", help="Custom header (format: Key:Value)")
    parser.add_argument("--output", help="Output JSON to file")

    args = parser.parse_args()

    proxies = {"http": args.proxy, "https": args.proxy} if args.proxy else None
    custom_headers = {}
    if args.header:
        for h in args.header:
            if ":" in h:
                key, val = h.split(":", 1)
                custom_headers[key.strip()] = val.strip()

    result = perform_full_recon(
        domain=args.domain,
        verbose=args.verbose,
        proxies=proxies,
        headers=custom_headers if custom_headers else None,
        scan=not args.no_scan
    )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[+] Output written to {args.output}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Recon interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"[!] Fatal error: {e}")
        sys.exit(1)
