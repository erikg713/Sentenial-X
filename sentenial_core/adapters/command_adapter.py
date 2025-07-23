from typing import List, Tuple
from sentenial_core.adapters.db_adapter import DBAdapter

class CommandAdapter:
    def __init__(self, db: DBAdapter):
        self._db = db

    async def get_pending(self, agent_id: str) -> List[Tuple[int, str]]:
        return await self._db.fetch_commands(agent_id)

    async def mark_done(self, cmd_id: int) -> None:
        await self._db.mark_command_processed(cmd_id)

    async def submit(self, agent_id: str, command: str) -> None:
        await self._db.enqueue_command(agent_id, command)
