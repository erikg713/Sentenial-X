import base64
import re

def normalize_input(payload: str) -> str:
    try:
        if re.match(r'^[A-Za-z0-9+/=]+$', payload):
            decoded = base64.b64decode(payload).decode('utf-8')
            return decoded
    except Exception:
        pass
    return payload
