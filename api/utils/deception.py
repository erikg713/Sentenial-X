# api/utils/deception.py
"""
Deception utilities for Sentenial-X.
Provides functions to generate honeypot data, fake responses, and misleading signals
to misdirect attackers during emulation or live defense operations.
"""

import random
import string
import logging
from datetime import datetime


logger = logging.getLogger("sentenialx.deception")


class DeceptionEngine:
    """
    Core engine for generating deceptive artifacts and signals.
    """

    @staticmethod
    def generate_fake_credentials() -> dict:
        """
        Generate a set of fake credentials for honeypot systems.
        """
        usernames = ["admin", "root", "sys", "operator", "guest"]
        username = random.choice(usernames)
        password = ''.join(random.choices(string.ascii_letters + string.digits, k=12))

        fake_creds = {
            "username": username,
            "password": password,
            "created_at": datetime.utcnow().isoformat()
        }

        logger.debug(f"Generated fake credentials: {fake_creds}")
        return fake_creds

    @staticmethod
    def generate_fake_filesystem_structure() -> dict:
        """
        Generate a fake filesystem tree for honeypot exploration.
        """
        structure = {
            "/home": ["admin", "user", "test"],
            "/etc": ["passwd", "shadow", "hosts"],
            "/var": ["log", "tmp", "www"],
            "/root": ["secrets.txt", "id_rsa", "notes.md"]
        }

        logger.debug("Generated fake filesystem structure")
        return structure

    @staticmethod
    def generate_fake_logs(num_entries: int = 10) -> list:
        """
        Generate fake log entries to confuse attackers.
        """
        levels = ["INFO", "WARNING", "ERROR", "DEBUG"]
        messages = [
            "User login successful",
            "Database connection established",
            "Service started successfully",
            "Unauthorized access attempt detected",
            "System resource threshold exceeded",
            "Kernel module loaded",
            "Configuration file updated"
        ]

        logs = []
        for _ in range(num_entries):
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": random.choice(levels),
                "message": random.choice(messages),
            }
            logs.append(log_entry)

        logger.debug(f"Generated {num_entries} fake logs")
        return logs

    @staticmethod
    def generate_network_noise(num_packets: int = 5) -> list:
        """
        Generate fake network packets for deception.
        """
        protocols = ["TCP", "UDP", "ICMP", "HTTP", "DNS"]
        packets = []
        for _ in range(num_packets):
            packet = {
                "src_ip": f"192.168.{random.randint(0, 255)}.{random.randint(1, 254)}",
                "dst_ip": f"10.0.{random.randint(0, 255)}.{random.randint(1, 254)}",
                "protocol": random.choice(protocols),
                "size_bytes": random.randint(40, 1500),
                "timestamp": datetime.utcnow().isoformat()
            }
            packets.append(packet)

        logger.debug(f"Generated {num_packets} fake network packets")
        return packets


# Example global engine
deception_engine = DeceptionEngine()
