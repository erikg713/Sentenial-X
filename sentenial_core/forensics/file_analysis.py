import os
import hashlib
from datetime import datetime

def compute_hash(file_path: str, algo: str = 'sha256') -> str:
    """
    Computes the hash of a file using the specified algorithm.
    """
    hash_func = getattr(hashlib, algo)()
    with open(file_path, 'rb') as f:
        while chunk := f.read(4096):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def get_file_metadata(file_path: str) -> dict:
    """
    Extracts basic metadata from a file.
    """
    if not os.path.exists(file_path):
        return {"error": "File does not exist"}

    stat = os.stat(file_path)
    return {
        "path": file_path,
        "size": stat.st_size,
        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
        "sha256": compute_hash(file_path),
    }