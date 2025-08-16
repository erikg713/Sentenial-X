"""
Sentenial-X Emulation Package
-----------------------------
Provides modules for threat emulation, including:
- Command & Control (C2) simulations
- Malware loader simulations
- Red-team exercises
"""

from .command_and_control import (
    AGENTS,
    Agent,
    register_agent,
    send_command,
    broadcast_command,
)
from .malware_loader import (
    MALWARE_PAYLOADS,
    load_and_execute_malware,
    load_multiple_payloads,
)

__all__ = [
    "AGENTS",
    "Agent",
    "register_agent",
    "send_command",
    "broadcast_command",
    "MALWARE_PAYLOADS",
    "load_and_execute_malware",
    "load_multiple_payloads",
]

# Versioning
__version__ = "1.0.0"

# Optional package-wide logger
import logging

logger = logging.getLogger("sentenial.emulation")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

logger.info("Sentenial-X Emulation package loaded.")