import os
import hashlib
from datetime import datetime

def collect_sample(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError("Sample not found.")

    with open(file_path, 'rb') as f:
        data = f.read()

    sample_hash = hashlib.sha256(data).hexdigest()
    sample_time = datetime.utcnow().isoformat()

    return {
        "hash": sample_hash,
        "collected_at": sample_time,
        "path": file_path,
        "size_kb": len(data) // 1024
    }
