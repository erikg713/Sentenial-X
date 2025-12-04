"""
core/cortex/signal_router.py

SignalRouter
-------------
Top-level orchestrator inside CORTEX.

It connects:
    - Brainstem          → Fast, reflex-level filtering
    - Analyzer (semantic)→ Deep meaning extraction
    - Engine (strategic) → Decision + action scoring

This router produces a unified bundle describing how the system
understands, contextualizes, and chooses actions for a given signal.
"""

from __future__ import annotations
from typing import Any, Dict


class SignalRouter:
    """
    High-level routing unit.
    Takes a raw signal and pushes it through:
        1. Brainstem      (fast filter / reflex)
        2. Analyzer       (semantic parsing / labeling)
        3. Engine         (strategic evaluation)
    """

    def __init__(self, brainstem, analyzer, engine):
        self.brainstem = brainstem          # core/cortex/brainstem.py
        self.analyzer = analyzer            # core/cortex/semantic_analyzer/*
        self.engine = engine                # core/cortex/engine.py

    def handle(self, signal: Any) -> Dict[str, Any]:
        """
        Process the signal through the 3 stages.
        Returns a structured bundle for downstream routers.
        """

        # --------------------
        # Step 1: Reflex layer
        # --------------------
        try:
            reflex = self.brainstem.process_signal(signal)
        except Exception as exc:
            reflex = {"error": str(exc)}

        # ---------------------------
        # Step 2: Semantic extraction
        # ---------------------------
        try:
            semantic_result = self.analyzer.analyze(signal)
        except Exception as exc:
            semantic_result = {"error": str(exc)}

        # -----------------------------
        # Step 3: Strategic evaluation
        # -----------------------------
        try:
            decision = self.engine.evaluate(signal, semantic_result)
        except Exception as exc:
            decision = {"error": str(exc)}

        return {
            "reflex": reflex,
            "semantic": semantic_result,
            "decision": decision,
        }
