# apps/dashboard/api/__init__.py
"""
API layer to fetch data from agents and provide to dashboard
"""
from .agents_api import fetch_agent_status, fetch_telemetry, fetch_threats, fetch_countermeasures
