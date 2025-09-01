"""
cli/orchestrator.py

Orchestrator module for Sentenial-X CLI.

Purpose:
- Execute central actions (policy updates, host isolation, blocks, rollbacks)
- Ensure all actions are logged via memory_adapter
- Provide structured, async results suitable for CLI output
"""

import asyncio
import json
from typing import Dict, Any, Optional
from cli.memory_adapter import get_adapter
from cli.logger import default_logger


# ------------------------------
# Orchestrator Core
# ------------------------------
class Orchestrator:
    """
    Handles orchestrator actions and logs them.
    """

    def __init__(self):
        self.memory = get_adapter()

    async def execute(self, action: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute an orchestrator action.

        Args:
            action (str): Action name (e.g., 'update_policy', 'block_indicator', 'rollback')
            params (dict, optional): Parameters for the action

        Returns:
            dict: Structured result with success/failure
        """
        params = params or {}
        result = {"action": action, "params": params, "status": "pending"}

        try:
            # Map actions to internal handlers
            handler_name = f"_handle_{action}"
            handler = getattr(self, handler_name, None)

            if handler is None:
                raise NotImplementedError(f"Orchestrator action '{action}' not implemented.")

            # Execute handler
            result_data = await handler(params)
            result.update({"status": "success", "result": result_data})

            # Log to memory
            await self.memory.log_command(action, params, result)
            default_logger.info(f"Orchestrator executed action '{action}' successfully.")

        except Exception as e:
            result.update({"status": "failed", "error": str(e)})
            await self.memory.log_command(action, params, result)
            default_logger.error(f"Failed orchestrator action '{action}': {e}")

        return result

    # --------------------------
    # Example Internal Handlers
    # --------------------------
    async def _handle_update_policy(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Simulate updating a policy
        policy_id = params.get("policy_id")
        mode = params.get("mode", "enforce")
        await asyncio.sleep(0.1)  # simulate async work
        return {"policy_id": policy_id, "mode": mode, "message": "Policy updated."}

    async def _handle_block_indicator(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Simulate blocking an indicator (IP, hash, etc.)
        indicator_type = params.get("type")
        value = params.get("value")
        ttl = params.get("ttl", "24h")
        await asyncio.sleep(0.1)
        return {"type": indicator_type, "value": value, "ttl": ttl, "message": "Indicator blocked."}

    async def _handle_isolate_host(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Simulate host isolation
        hostname = params.get("hostname")
        await asyncio.sleep(0.1)
        return {"hostname": hostname, "message": "Host isolated."}

    async def _handle_rollback(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Simulate rollback
        change_id = params.get("change_id")
        await asyncio.sleep(0.1)
        return {"change_id": change_id, "message": "Rollback executed."}


# ------------------------------
# Singleton instance helper
# ------------------------------
_orchestrator_instance: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    """Return singleton orchestrator instance."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = Orchestrator()
    return _orchestrator_instance


# ------------------------------
# Quick CLI Test
# ------------------------------
if __name__ == "__main__":
    async def test_orchestrator():
        orch = get_orchestrator()

        # Test update_policy
        res1 = await orch.execute("update_policy", {"policy_id": "123", "mode": "enforce"})
        print("Update policy:", res1)

        # Test block_indicator
        res2 = await orch.execute("block_indicator", {"type": "ip", "value": "203.0.113.42"})
        print("Block indicator:", res2)

        # Test isolate_host
        res3 = await orch.execute("isolate_host", {"hostname": "db-02"})
        print("Isolate host:", res3)

        # Test rollback
        res4 = await orch.execute("rollback", {"change_id": "chg-20250822-001"})
        print("Rollback:", res4)

        # Test unknown action
        res5 = await orch.execute("unknown_action", {})
        print("Unknown action:", res5)

    asyncio.run(test_orchestrator())
