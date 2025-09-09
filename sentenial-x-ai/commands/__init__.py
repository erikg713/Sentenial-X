"""
Sentenial-X CLI Commands
------------------------
Subcommand entrypoints for `sentenialx` CLI.
"""

from . import train, eval, serve, telemetry

__all__ = ["train", "eval", "serve", "telemetry"]
