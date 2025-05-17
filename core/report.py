# core/report.py

import os
import json
from datetime import datetime

def save_report(data, report_type="recon"):
    os.makedirs("reports", exist_ok=True)
    filename = f"reports/{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    return filename
