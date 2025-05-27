import re
import ipaddress
from datetime import datetime
from typing import Dict, Union, Optional, List, Tuple

# Tor exit node detection - Common patterns
TOR_PATTERNS = [
    r"\.onion$",                      # Onion services
    r"^185\.220\.(100|101|102)\.\d+$",  # Tor exit node range
    r"^154\.(\d+)\.(\d+)\.(\d+)$",    # Common Tor exit nodes
    r"^51\.(\d+)\.(\d+)\.(\d+)$",     # Additional Tor nodes
]

# Pre-compile regexes for better performance
TOR_REGEX = [re.compile(pattern) for pattern in TOR_PATTERNS]

# Known Tor exit node subnet blocks - manually maintained list
TOR_SUBNET_BLOCKS = [
    "185.220.100.0/22",
    "51.15.0.0/16", 
    "154.35.0.0/16"
]

def is_ip_in_subnet(ip: str, subnet_blocks: List[str]) -> bool:
    """Check if IP is in any of the given subnet blocks."""
    # Skip expensive check if not IP-like
    if not re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", ip):
        return False
    
    try:
        ip_obj = ipaddress.ip_address(ip)
        return any(ip_obj in ipaddress.ip_network(block) for block in subnet_blocks)
    except ValueError:
        # Not a valid IP address
        return False

def run_detection(target: str) -> Dict[str, Union[str, Dict]]:
    """
    Enhanced Tor network detection function.
    
    Args:
        target: URL, domain or IP address to check
        
    Returns:
        Dictionary with detection status and details
    """
    # Input validation
    if not target or not isinstance(target, str):
        return {
            "status": "error", 
            "message": f"Invalid target: {target}"
        }
    
    # Normalize target for consistent checking
    target = target.lower().strip()
    
    # Check for Tor patterns
    detected = False
    detection_method = None
    
    # Quick regex pattern check (faster than subnet check)
    for idx, pattern in enumerate(TOR_REGEX):
        if pattern.search(target):
            detected = True
            detection_method = f"pattern_match_{idx}"
            break
    
    # More thorough subnet check if it looks like an IP
    if not detected and is_ip_in_subnet(target, TOR_SUBNET_BLOCKS):
        detected = True
        detection_method = "subnet_match"
    
    # Build result with additional metadata
    if detected:
        return {
            "status": "stealth",
            "message": "Tor network access detected",
            "details": {
                "target": target,
                "detection_method": detection_method,
                "severity": "medium",
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "confidence": 0.85 if detection_method == "subnet_match" else 0.95
            }
        }
    
    # No Tor traffic found
    return {
        "status": "clear",
        "message": "No Tor traffic found",
        "details": {
            "target": target,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        }
    }

# Example usage
if __name__ == "__main__":
    # Test cases
    targets = [
        "example.com",
        "suspicious.onion",
        "185.220.101.45",
        "154.35.175.225",
        "51.15.43.232"
    ]
    
    for t in targets:
        result = run_detection(t)
        print(f"Target: {t} -> {result['status']}")
