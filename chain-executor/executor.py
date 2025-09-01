# chain-executor/executor.py

import asyncio
import logging
import importlib
import traceback
from pathlib import Path
from typing import Dict, Any, Callable, Awaitable, List, Optional

from chain_executor.utils import load_config, validate_chain

logger = logging.getLogger("chain-executor")

class ChainExecutor:
    """
    Executes a chain of modular tasks dynamically.
    Each task in the chain is defined by a Python module and function reference.
    """

    def __init__(self, config_path: str = "chain-executor/config.yaml"):
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)
        self.chains = self.config.get("chains", {})
        self.context: Dict[str, Any] = {}
        logger.debug(f"ChainExecutor initialized with {len(self.chains)} chains")

    async def execute_task(
        self, 
        module_name: str, 
        function_name: str, 
        params: Dict[str, Any]
    ) -> Any:
        """
        Dynamically import a module and execute a coroutine or function.
        """
        try:
            logger.debug(f"Loading task: {module_name}.{function_name} with params {params}")
            module = importlib.import_module(module_name)
            func: Callable = getattr(module, function_name)

            if asyncio.iscoroutinefunction(func):
                result = await func(**params)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: func(**params))

            logger.info(f"Task {module_name}.{function_name} executed successfully.")
            return result
        except Exception as e:
            logger.error(f"Error executing task {module_name}.{function_name}: {e}")
            traceback.print_exc()
            return {"error": str(e)}

    async def execute_chain(self, chain_name: str, initial_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a predefined chain of tasks.
        """
        if chain_name not in self.chains:
            raise ValueError(f"Chain '{chain_name}' not found in config.")

        chain = self.chains[chain_name]
        validate_chain(chain)

        self.context = initial_context or {}
        logger.info(f"Executing chain: {chain_name}")

        for step in chain:
            module = step.get("module")
            function = step.get("function")
            params = step.get("params", {})

            # Merge context into params
            merged_params = {**self.context, **params}

            logger.debug(f"Executing step: {module}.{function} with params: {merged_params}")
            result = await self.execute_task(module, function, merged_params)

            # Store result in context
            self.context[function] = result
            logger.debug(f"Step result stored in context: {function} -> {result}")

        logger.info(f"Chain {chain_name} execution completed.")
        return self.context

    async def run(self):
        """
        Run all chains defined in config sequentially.
        """
        logger.info("Starting execution of all configured chains...")
        results: Dict[str, Dict[str, Any]] = {}
        for chain_name in self.chains.keys():
            results[chain_name] = await self.execute_chain(chain_name)
        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a chain executor workflow")
    parser.add_argument("--config", type=str, default="chain-executor/config.yaml", help="Path to config file")
    parser.add_argument("--chain", type=str, required=False, help="Specific chain to run")

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

    executor = ChainExecutor(config_path=args.config)

    async def main():
        if args.chain:
            result = await executor.execute_chain(args.chain)
            print(f"Chain {args.chain} result:\n{result}")
        else:
            result = await executor.run()
            print("All chain results:\n", result)

    asyncio.run(main())
