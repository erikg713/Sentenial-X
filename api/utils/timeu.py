# api/utils/timeu.py

"""
Time Utilities Module
---------------------

This module provides standardized time utility functions for:
- UTC datetime handling
- Duration formatting
- Execution timers
- Timestamp parsing and validation

These utilities ensure consistency across all API services, logging,
and risk analysis workflows.

Author: Sentenial-X Team
"""

import time
import datetime
from typing import Optional, Union


def utcnow() -> datetime.datetime:
    """
    Get the current UTC datetime with timezone awareness.

    Returns:
        datetime: Current UTC time with tzinfo.
    """
    return datetime.datetime.now(datetime.timezone.utc)


def format_timestamp(ts: Optional[datetime.datetime] = None) -> str:
    """
    Format a datetime object into a standardized ISO8601 UTC string.

    Args:
        ts (datetime, optional): The datetime to format. Defaults to now().

    Returns:
        str: ISO8601 formatted timestamp.
    """
    ts = ts or utcnow()
    return ts.astimezone(datetime.timezone.utc).isoformat(timespec="seconds")


def parse_timestamp(ts_str: str) -> datetime.datetime:
    """
    Parse an ISO8601 timestamp string into a datetime object.

    Args:
        ts_str (str): ISO8601 string.

    Returns:
        datetime: Parsed datetime object (UTC).
    """
    try:
        dt = datetime.datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.astimezone(datetime.timezone.utc)
    except Exception as e:
        raise ValueError(f"Invalid timestamp format: {ts_str}") from e


def format_duration(seconds: Union[int, float]) -> str:
    """
    Convert seconds into human-readable duration.

    Args:
        seconds (int | float): Duration in seconds.

    Returns:
        str: Formatted string like '1h 23m 45s'.
    """
    seconds = int(seconds)
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{sec}s")

    return " ".join(parts)


class Timer:
    """
    Simple performance timer for measuring code execution duration.
    """

    def __init__(self, label: str = "Timer"):
        self.label = label
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start(self):
        """Start the timer."""
        self.start_time = time.perf_counter()
        self.end_time = None

    def stop(self) -> float:
        """
        Stop the timer and return the elapsed duration.

        Returns:
            float: Elapsed time in seconds.
        """
        if self.start_time is None:
            raise RuntimeError("Timer was not started.")
        self.end_time = time.perf_counter()
        return self.elapsed

    @property
    def elapsed(self) -> float:
        """
        Get the elapsed time in seconds.

        Returns:
            float: Elapsed duration.
        """
        if self.start_time is None:
            return 0.0
        end_time = self.end_time or time.perf_counter()
        return end_time - self.start_time

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        if exc_type is None:
            print(f"[{self.label}] completed in {format_duration(self.elapsed)}.")


# Quick self-test
if __name__ == "__main__":
    print("UTC Now:", format_timestamp())
    print("Parsed:", parse_timestamp("2025-09-02T12:34:56+00:00"))
    print("Duration:", format_duration(3671))  # -> "1h 1m 11s"

    with Timer("Example Task") as t:
        time.sleep(1.2)
