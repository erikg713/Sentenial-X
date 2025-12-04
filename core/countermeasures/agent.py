#!/usr/bin/env python3
"""
Countermeasure Agent
Executes safe, policy-compliant actions based on threat insights.
"""

from typing import Callable, Dict, Any
from .policy import PolicyEngine
from .sandbox import SandboxExecutor
from .logger import ActionLogger

class CountermeasureAgent:
    def __init__(self, policy_engine: PolicyEngine = None, sandbox: SandboxExecutor = None, logger: ActionLogger = None):
        self.policy_engine = policy_engine or PolicyEngine()
        self.sandbox = sandbox or SandboxExecutor()
        self.logger = logger or ActionLogger()

    def execute_action(self, action_name: str, action_callable: Callable, user_role: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a countermeasure action with policy validation, sandboxing, and logging
        """
        # 1️⃣ Policy validation
        if not self.policy_engine.is_authorized(user_role, action_name):
            result = {"status": "unauthorized"}
        else:
            # 2️⃣ Sandbox execution
            result = self.sandbox.execute(action_callable, context)

        # 3️⃣ Logging
        log_path = self.logger.log(action_name, user_role, result)
        result["log_path"] = log_path
        return result
