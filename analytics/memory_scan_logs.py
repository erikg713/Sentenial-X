import os
from datetime import datetime

LOG_DIR = "analytics/memory_scan_logs"
LOG_FILE = os.path.join(LOG_DIR, f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

os.makedirs(LOG_DIR, exist_ok=True)

def log_scan_result(scan_data: dict):
    """Append scan result data to a timestamped log file."""
    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.now()}] {scan_data}\n")


def read_latest_logs(n: int = 10):
    """Read the last `n` entries from the latest scan log file."""
    try:
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
            return lines[-n:]
    except FileNotFoundError:
        return ["No log file found."]


# Example usage:
if __name__ == "__main__":
    log_scan_result({"module": "IDS", "status": "clean", "ip": "192.168.1.10"})
    print("Last 5 logs:")
    for line in read_latest_logs(5):
        print(line.strip())
