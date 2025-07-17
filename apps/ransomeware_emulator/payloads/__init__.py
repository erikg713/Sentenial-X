"""
Payload registry for simulated ransomware actions.
Each function here simulates a type of ransomware behavior.

To register new payloads, define them in this folder
and import them below for easy access in sandbox runner.
"""

from .sample_encrypt import sample_encrypt_payload

__all__ = [
    "sample_encrypt_payload"
]

