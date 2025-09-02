
"""
Custom Scanners Plugin Package for Sentenial-X

Provides advanced scanning modules for monitoring malware, network,
and system anomalies. Designed for integration with the Sentenial-X
threat detection platform.

Modules:
- base_scanner       : Abstract base class for all scanners
- malware_scanner    : Scans for malware signatures and anomalies
- network_scanner    : Scans for suspicious network activity
- system_scanner     : Scans for system-level irregularities
- utils              : Helper functions for scanning modules
"""

from .base_scanner import BaseScanner
from .malware_scanner import MalwareScanner
from .network_scanner import NetworkScanner
from .system_scanner import SystemScanner

__all__ = [
    "BaseScanner",
    "MalwareScanner",
    "NetworkScanner",
    "SystemScanner",
]
