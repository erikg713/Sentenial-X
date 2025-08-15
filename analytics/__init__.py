# sentenial-x/analytics/__init__.py
"""
Sentenial-X Analytics Module
----------------------------
Centralized analytics and monitoring for Sentenial-X.

Responsibilities:
- Collect and aggregate agent telemetry and logs.
- Perform statistical and ML-driven analysis.
- Generate threat summaries and risk scores.
- Provide interfaces for dashboards and reports.
"""

from .telemetry_aggregator import TelemetryAggregator
from .threat_metrics import ThreatMetrics
from .reporting import ReportGenerator
from .memory_scan_logs import MemoryScanLogs
