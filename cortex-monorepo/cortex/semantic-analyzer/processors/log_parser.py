import re
from typing import Dict
from models.base import SemanticEvent

class LogParser:
    """
    Parses raw log strings into structured SemanticEvent objects.
    """
    @staticmethod
    def parse_log(log: str, source: str) -> SemanticEvent:
        event_id = f"{source}-{hash(log)}"
        timestamp = ""  # ideally extract from log if available
        raw_data = {"log": log}

        return SemanticEvent(
            event_id=event_id,
            source=source,
            timestamp=timestamp,
            raw_data=raw_data
        )
