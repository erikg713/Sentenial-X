"""
SIEM Integration Utilities
--------------------------
This module provides utilities to integrate with external SIEM platforms
(Splunk, Elastic, QRadar, etc.), enabling centralized log and event management.

Author: Sentenial-X Team
"""

import json
import socket
import logging
import queue
import threading
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger("siem_utils")


class SIEMClient:
    """
    A client for sending structured security events to SIEM platforms.
    Supports syslog over UDP/TCP and direct HTTP API integrations (future-ready).
    """

    def __init__(self, host: str, port: int, protocol: str = "udp", facility: int = 1):
        self.host = host
        self.port = port
        self.protocol = protocol.lower()
        self.facility = facility
        self._socket = None
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._process_queue, daemon=True)

        if self.protocol not in ("udp", "tcp"):
            raise ValueError("Protocol must be 'udp' or 'tcp'")

        self._connect()
        self._thread.start()
        logger.info(f"SIEMClient initialized for {self.protocol.upper()} {self.host}:{self.port}")

    def _connect(self):
        """Establish socket connection based on protocol."""
        if self.protocol == "udp":
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        else:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.connect((self.host, self.port))

    def _process_queue(self):
        """Background thread for processing queued messages."""
        while not self._stop_event.is_set():
            try:
                message = self._queue.get(timeout=1)
                self._send_now(message)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Failed to send SIEM message: {e}")

    def _send_now(self, message: str):
        """Send a message immediately over the socket."""
        try:
            if self.protocol == "udp":
                self._socket.sendto(message.encode("utf-8"), (self.host, self.port))
            else:
                self._socket.sendall(message.encode("utf-8") + b"\n")
        except Exception as e:
            logger.error(f"Error sending SIEM message: {e}")

    def send_event(self, event: Dict, severity: str = "INFO"):
        """
        Send an event to the SIEM system.
        :param event: Dictionary of structured event data.
        :param severity: Log level (INFO, WARNING, ERROR, CRITICAL).
        """
        envelope = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "severity": severity,
            "event": event,
            "facility": self.facility,
        }
        message = json.dumps(envelope)
        self._queue.put(message)

    def close(self):
        """Shut down the client gracefully."""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join()
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
        logger.info("SIEMClient closed")


# Global convenience wrapper
_siem_client: Optional[SIEMClient] = None


def init_siem(host: str, port: int, protocol: str = "udp", facility: int = 1):
    """Initialize global SIEM client."""
    global _siem_client
    _siem_client = SIEMClient(host, port, protocol, facility)


def log_to_siem(event: Dict, severity: str = "INFO"):
    """Send a structured event to the SIEM via the global client."""
    if not _siem_client:
        logger.warning("SIEM client not initialized, dropping event")
        return
    _siem_client.send_event(event, severity)


def shutdown_siem():
    """Close the global SIEM client."""
    global _siem_client
    if _siem_client:
        _siem_client.close()
        _siem_client = None
