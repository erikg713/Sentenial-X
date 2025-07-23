from typing import Dict, Any
from sentenial_core.adapters.db_adapter import DBAdapter

class LearnerAdapter:
    """
    Wraps memory fetching and simple statistics computation.
    """

    def __init__(self, db: DBAdapter):
        self._db = db

    async def learn(self) -> Dict[str, int]:
        """
        Count how many times each command has occurred in memory.
        """
        stats: Dict[str, int] = {}
        events = await self._db.fetch_memory()

        for ev in events:
            cmd = ev.get("command")
            if not cmd:
                continue
            stats[cmd] = stats.get(cmd, 0) + 1

        return stats
