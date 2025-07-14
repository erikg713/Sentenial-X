"""
Sentenial X :: Core Cortex Module

This package contains intelligence orchestration logic, including:
- Behavioral correlation
- Threat scoring
- Anomaly detection
- Signature & IOC fusion
- Event context enrichment

Modules:
    - cortex_manager.py: Main interface for coordinating analysis pipelines.
    - analyzer.py: Behavioral analytics engine.
    - fusion_engine.py: IOC/signature matcher and score calculator.
    - enrichment.py: Metadata and telemetry enricher.

Usage:
    from core.cortex import CortexManager

    cortex = CortexManager()
    result = cortex.analyze(session_id="abc123", telemetry_stream=...)
"""

from .cortex_manager import CortexManager

