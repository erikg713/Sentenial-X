import logging
from typing import List, Dict, Callable, Any

logger = logging.getLogger("ChainExecutor")
logging.basicConfig(level=logging.INFO)

class ChainExecutionError(Exception):
    """Custom exception for chain execution failures."""
    pass

class ChainExecutor:
    def __init__(self, steps: List[Dict[str, Any]]):
        """
        Initialize with a list of steps. Each step is a dict:
        {
            "name": "Step Name",
            "action": Callable[[Dict], Dict],
            "input_keys": [...],   # optional: keys required from context
            "output_keys": [...]   # optional: keys to extract from result into context
        }
        """
        self.steps = steps
        self.context: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        """Executes the chain and returns the final context."""
        for i, step in enumerate(self.steps):
            name = step.get("name", f"Step-{i}")
            action = step["action"]
            input_keys = step.get("input_keys", [])
            output_keys = step.get("output_keys", [])

            # Prepare inputs from context
            inputs = {k: self.context.get(k) for k in input_keys}

            logger.info(f"[{name}] Executing with inputs: {inputs}")

            try:
                result = action(inputs)
                logger.info(f"[{name}] Result: {result}")
            except Exception as e:
                logger.error(f"[{name}] Failed: {e}")
                raise ChainExecutionError(f"Step '{name}' failed") from e

            # Store selected outputs into context
            for key in output_keys:
                if key in result:
                    self.context[key] = result[key]

        logger.info("Chain execution complete.")
        return self.context

