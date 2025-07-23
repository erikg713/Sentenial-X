import asyncio
from typing import Callable, Dict, Optional

class ExecutorAdapter:
    """
    Central registry and executor for all commands (built-in + plugins).
    """

    _commands: Dict[str, Callable[[str], Optional[str]]] = {}

    @classmethod
    def register(cls, name: str, fn: Callable[[str], Optional[str]]) -> None:
        """
        Register a new command handler under `name`.
        """
        cls._commands[name] = fn

    @classmethod
    async def execute(cls, cmd: str) -> Optional[str]:
        """
        Split the incoming text into command + args, invoke the handler, and return its result.
        """
        name, *parts = cmd.strip().split(" ", 1)
        arg = parts[0] if parts else ""

        # Plugin / custom command
        if name in cls._commands:
            # run sync function in a threadpool to avoid blocking
            return await asyncio.to_thread(cls._commands[name], arg)

        # Built-in logic
        if name == "shutdown":
            # propagate SystemExit to trigger graceful shutdown
            raise SystemExit("shutdown requested")
        if name == "echo":
            return arg

        # Unknown commands return None
        return None
