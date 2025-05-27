# core/ids/tor_exit_node_ids.py

def run_detection(target):
    # Simulate Tor exit node check
    known_tor_ips = ["10.0.0.5", "185.220.101.1", "104.244.72.115"]
    if target in known_tor_ips:
        return {
            "status": "anonymized",
            "message": "Connection from Tor exit node."
        }
    else:
        return {
            "status": "clear",
            "message": "Not a known Tor node."
        }
