"""
Adds additional metadata to raw telemetry events.
"""

from typing import Dict, Any
import geoip2.database
import socket

GEOIP_DB_PATH = "data/GeoLite2-City.mmdb"

def enrich_event(event: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(event)
    ip = event.get("remote_ip")
    if ip:
        try:
            with geoip2.database.Reader(GEOIP_DB_PATH) as reader:
                response = reader.city(ip)
                enriched["geoip"] = {
                    "city": response.city.name,
                    "country": response.country.name,
                    "lat": response.location.latitude,
                    "lon": response.location.longitude
                }
        except Exception:
            enriched["geoip"] = {"error": "lookup_failed"}

    pid = event.get("pid")
    if pid:
        enriched["process_context"] = f"pid:{pid} user:{event.get('user', 'unknown')}"
    return enriched

