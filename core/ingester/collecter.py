# core/ingestor/collector.py
import os
import hashlib
from datetime import datetime 

def collect_sample(file_path, initial_bytes_length=128, first_lines_count=10):
    sample_info = {
        "file_name": os.path.basename(file_path),
        "file_size": os.path.getsize(file_path),
        "file_path": file_path
    }

    md5_hash = hashlib.md5()
    sha256_hash = hashlib.sha256()

    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(4096):
                md5_hash.update(chunk)
                sha256_hash.update(chunk)
        sample_info["md5"] = md5_hash.hexdigest()
        sample_info["sha256"] = sha256_hash.hexdigest()
    except Exception as e:
        sample_info["hashing_error"] = str(e)

    try:
        with open(file_path, 'rb') as f:
            initial_bytes = f.read(initial_bytes_length)
            sample_info["initial_bytes_hex"] = initial_bytes.hex()
    except Exception as e:
        sample_info["initial_bytes_error"] = str(e)

    first_lines = []
    try:
        with open(file_path, 'r', errors='ignore') as f:
            for _ in range(first_lines_count):
                line = f.readline()
                if not line:
                    break
                first_lines.append(line.strip())
        sample_info["first_lines"] = first_lines
    except Exception as e:
        sample_info["reading_error"] = str(e)

    return sample_info
