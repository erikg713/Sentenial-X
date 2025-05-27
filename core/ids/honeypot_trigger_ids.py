# core/ids/honeypot_trigger_ids.py

def run_detection(target):
    # Fake honeypot interaction log
    interaction_log = [
        {"ip": "127.0.0.1", "triggered": False},
        {"ip": "192.168.1.10", "triggered": True}
    ]

    entry = next((log for log in interaction_log if log["ip"] == target), None)
    if entry and entry["triggered"]:
        return {
            "status": "triggered",
            "message": f"Honeypot triggered by {target}"
        }
    else:
        return {
            "status": "clear",
            "message": "No honeypot interactions."
        }
