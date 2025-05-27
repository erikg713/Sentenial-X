def run_detection(target):
    if ".onion" in target or "185.220" in target:
        return {"status": "stealth", "message": "Tor network access detected"}
    return {"status": "clear", "message": "No Tor traffic found"}
