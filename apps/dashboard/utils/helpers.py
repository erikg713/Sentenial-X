# apps/dashboard/utils/helpers.py
from datetime import datetime

def current_utc_iso():
    return datetime.utcnow().isoformat() + "Z"
