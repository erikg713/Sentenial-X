import psutil
from datetime import datetime

class NetworkWatcher:
    def __init__(self):
        self.known_connections = set()

    def scan(self):
        events = []
        for conn in psutil.net_connections(kind="inet"):
            try:
                laddr = f"{conn.laddr.ip}:{conn.laddr.port}"
                raddr = f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else None
                conn_key = (laddr, raddr, conn.status)
                if conn_key not in self.known_connections:
                    self.known_connections.add(conn_key)
                    if raddr:
                        events.append(f"New connection to {raddr} from {laddr} (Status: {conn.status})")
            except Exception:
                continue
        return events