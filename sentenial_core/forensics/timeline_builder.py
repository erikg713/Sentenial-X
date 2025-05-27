import re
import ipaddress
from datetime import datetime
from typing import Dict, Union, List

# Tor exit node detection patterns
TOR_PATTERNS = [
    r"\.onion$",                            # Onion services
    r"^185\.220\.(100|101|102)\.\d+$",     # Tor exit node range
    r"^154\.\d+\.\d+\.\d+$",               # Common Tor exit nodes
    r"^51\.\d+\.\d+\.\d+$",                # Additional Tor nodes
]

# Pre-compiled regex patterns for performance
TOR_REGEX = [re.compile(pattern) for pattern in TOR_PATTERNS]

# Known Tor exit node subnets (manually curated)
TOR_SUBNET_BLOCKS = [
    "185.220.100.0/22",
    "51.15.0.0/16", 
    "154.35.0.0/16"
]

def is_ip_in_subnet(ip: str, subnet_blocks: List[str]) -> bool:
    """Check if an IP address belongs to any known Tor subnet."""
    if not re.match(r"^\d{1,3}(\.\d{1,3}){3}$", ip):
        return False
    try:
        ip_obj = ipaddress.ip_address(ip)
        return any(ip_obj in ipaddress.ip_network(block) for block in subnet_blocks)
    except ValueError:
        return False

def run_detection(target: str) -> Dict[str, Union[str, Dict]]:
    """
    Detects if a given domain, URL, or IP address is linked to the Tor network.
    
    Args:
        target: The domain name, URL, or IP address to analyze.
        
    Returns:
        A dictionary indicating detection status and relevant metadata.
    """
    if not target or not isinstance(target, str):
        return {
            "status": "error", 
            "message": f"Invalid target: {target}"
        }

    target = target.lower().strip()
    detected = False
    detection_method = None

    # Fast pattern match
    for idx, pattern in enumerate(TOR_REGEX):
        if pattern.search(target):
            detected = True
            detection_method = f"pattern_match_{idx}"
            break

    # Fallback to subnet verification if not already detected
    if not detected and is_ip_in_subnet(target, TOR_SUBNET_BLOCKS):
        detected = True
        detection_method = "subnet_match"

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    if detected:
        return {
            "status": "stealth",
            "message": "Tor network access detected",
            "details": {
                "target": target,
                "detection_method": detection_method,
                "severity": "medium",
                "timestamp": timestamp,
                "confidence": 0.85 if detection_method == "subnet_match" else 0.95
            }
        }

    return {
        "status": "clear",
        "message": "No Tor traffic found",
        "details": {
            "target": target,
            "timestamp": timestamp
        }
    }

if __name__ == "__main__":
    test_targets = [
        "example.com",
        "suspicious.onion",
        "185.220.101.45",
        "154.35.175.225",
        "51.15.43.232"
    ]

    for target in test_targets:
        result = run_detection(target)
        print(f"Target: {target} -> {result['status']}")