# modules/recon/osint.py

import socket
import requests
import logging

logger = logging.getLogger("sentenialx.recon.osint")
logging.basicConfig(level=logging.INFO)

class OSINTScanner:
    def __init__(self):
        self.shodan_api_key = None
        self.abuseipdb_key = None

    def set_api_keys(self, shodan=None, abuseipdb=None):
        self.shodan_api_key = shodan
        self.abuseipdb_key = abuseipdb

    def basic_lookup(self, target):
        """Performs basic DNS resolution and WHOIS lookup."""
        logger.info(f"Performing basic OSINT lookup for {target}")
        result = {
            "hostname": target,
            "ip": "",
            "fqdn": "",
        }

        try:
            ip = socket.gethostbyname(target)
            fqdn = socket.getfqdn(target)
            result["ip"] = ip
            result["fqdn"] = fqdn
        except Exception as e:
            logger.error(f"DNS lookup failed: {e}")
        
        return result

    def shodan_lookup(self, ip):
        """Queries Shodan API for host data."""
        if not self.shodan_api_key:
            logger.warning("Shodan API key not configured.")
            return {}

        logger.info(f"Querying Shodan for IP: {ip}")
        try:
            url = f"https://api.shodan.io/shodan/host/{ip}?key={self.shodan_api_key}"
            res = requests.get(url)
            if res.status_code == 200:
                return res.json()
            else:
                logger.error(f"Shodan error {res.status_code}: {res.text}")
        except Exception as e:
            logger.error(f"Shodan request failed: {e}")
        return {}

    def abuseipdb_lookup(self, ip):
        """Checks IP reputation from AbuseIPDB."""
        if not self.abuseipdb_key:
            logger.warning("AbuseIPDB API key not configured.")
            return {}

        logger.info(f"Querying AbuseIPDB for IP: {ip}")
        try:
            url = "https://api.abuseipdb.com/api/v2/check"
            headers = {"Key": self.abuseipdb_key, "Accept": "application/json"}
            params = {"ipAddress": ip, "maxAgeInDays": 90}
            res = requests.get(url, headers=headers, params=params)
            if res.status_code == 200:
                return res.json().get("data", {})
            else:
                logger.error(f"AbuseIPDB error {res.status_code}: {res.text}")
        except Exception as e:
            logger.error(f"AbuseIPDB request failed: {e}")
        return {}

    def full_scan(self, target):
        """Performs complete OSINT sweep."""
        logger.info(f"Starting full OSINT scan on {target}")
        result = self.basic_lookup(target)
        ip = result.get("ip")
        if ip:
            result["shodan"] = self.shodan_lookup(ip)
            result["abuseipdb"] = self.abuseipdb_lookup(ip)
        return result


# Example usage
if __name__ == "__main__":
    osint = OSINTScanner()
    osint.set_api_keys(
        shodan="YOUR_SHODAN_API_KEY",
        abuseipdb="YOUR_ABUSEIPDB_API_KEY"
    )
    target = "scanme.shodan.io"
    result = osint.full_scan(target)
    print("[+] OSINT Result:")
    print(result)

