# -*- coding: utf-8 -*-
"""
core.simulator.attack_playbook
------------------------------

Defines reusable attack playbooks for simulations.
Each playbook is a structured set of steps (actions, expected results, and metadata)
that can be used by emulation engines.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class AttackStep:
    """A single step in an attack scenario."""
    id: str
    description: str
    action: Callable[..., Any]
    expected_outcome: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class AttackPlaybook:
    """A complete attack scenario composed of multiple steps."""
    id: str
    name: str
    description: str
    steps: List[AttackStep]

    def run(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute all steps sequentially. A shared `context` dict can carry state across steps.
        """
        results: Dict[str, Any] = {}
        ctx = context or {}

        logger.info("Running playbook: %s (%s)", self.name, self.id)

        for step in self.steps:
            try:
                logger.debug("Executing step: %s - %s", step.id, step.description)
                output = step.action(ctx)
                results[step.id] = output
                if step.expected_outcome:
                    logger.debug("Step %s expected outcome: %s", step.id, step.expected_outcome)
            except Exception as exc:
                logger.exception("Step %s failed: %s", step.id, exc)
                results[step.id] = {"error": str(exc)}
                break  # stop execution on failure for realism
        return results


# Example factory
def create_playbook() -> AttackPlaybook:
    """Return a sample attack playbook with dummy steps."""

    def reconnaissance(ctx: Dict[str, Any]) -> str:
        ctx["recon"] = "open ports: 22, 80, 443"
        return ctx["recon"]

    def exploit(ctx: Dict[str, Any]) -> str:
        if "recon" not in ctx:
            raise RuntimeError("Recon not performed")
        ctx["exploit"] = "simulated RCE exploit"
        return ctx["exploit"]

    steps = [
        AttackStep(id="recon", description="Perform reconnaissance", action=reconnaissance),
        AttackStep(id="exploit", description="Exploit discovered service", action=exploit),
    ]

    return AttackPlaybook(
        id="pb-001",
        name="Sample Exploit Chain",
        description="A simple 2-step exploit chain: recon + exploit.",
        steps=steps,
    )
