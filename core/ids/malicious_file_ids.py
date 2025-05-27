# core/ids/malicious_file_ids.py

def run_detection(target):
    # Simulated hash match
    suspicious_hashes = ["e99a18c428cb38d5f260853678922e03", "d41d8cd98f00b204e9800998ecf8427e"]
    uploaded_hashes = [target]  # assuming the 'target' is the hash in this case

    flagged = [h for h in uploaded_hashes if h in suspicious_hashes]

    if flagged:
        return {
            "status": "infected",
            "message": f"Known malicious file hash detected: {flagged}"
        }
    else:
        return {
            "status": "clean",
            "message": "No malicious file hashes found."
        }
