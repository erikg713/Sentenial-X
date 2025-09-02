# api/utils/enrich.py

import socket
import requests
from datetime import datetime
from typing import Dict, Any, Optional


class EnrichmentService:
    """
    Provides enrichment utilities for telemetry and threat data.
    Includes IP geo-location, hostname resolution, and threat intelligence lookups.
    """

    def __init__(self, geoip_url: Optional[str] = None, threat_api_key: Optional[str] = None):
        self.geoip_url = geoip_url or "https://ipapi.co/{ip}/json/"
        self.threat_api_key = threat_api_key

    def resolve_hostname(self, ip: str) -> Optional[str]:
        """Resolve an IP address to a hostname."""
        try:
            return socket.gethostbyaddr(ip)[0]
        except Exception:
            return None

    def geoip_lookup(self, ip: str) -> Dict[str, Any]:
        """Get GeoIP data for an IP address."""
        try:
            url = self.geoip_url.format(ip=ip)
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"error": f"GeoIP lookup failed with status {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def threat_lookup(self, ip: str) -> Dict[str, Any]:
        """Perform a threat intelligence lookup for an IP address."""
        if not self.threat_api_key:
            return {"warning": "Threat API key not configured"}
        try:
            # Example threat intel API (replace with real provider)
            url = f"https://threat-intel.example.com/api/ip/{ip}"
            headers = {"Authorization": f"Bearer {self.threat_api_key}"}
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"error": f"Threat lookup failed with status {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def enrich_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a telemetry event with hostname, geoIP, and threat data.
        Expects an event dict with at least an 'ip' field.
        """
        enriched = event.copy()
        ip = event.get("ip")

        if not ip:
            enriched["enrichment"] = {"error": "No IP provided"}
            return enriched

        enriched["enrichment"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "hostname": self.resolve_hostname(ip),
            "geoip": self.geoip_lookup(ip),
            "threat_intel": self.threat_lookup(ip),
        }

        return enriched
