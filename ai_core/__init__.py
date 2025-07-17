"""
Sentenial X A.I. Core

This module contains the intelligence engine that powers threat detection,
feedback learning, secure logging, and runtime communication.
"""

from .detection import detect_prompt_threat
from .logger import log_threat_event
from .feedback import update_model
from .ipc_server import start_ipc_server

