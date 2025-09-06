# -*- coding: utf-8 -*-
"""
API Routes Package Initializer for Sentenial-X
----------------------------------------------

Imports and registers all route modules for the FastAPI application.
"""

from __future__ import annotations

from fastapi import APIRouter

from . import (
    alerts,
    cortex,
    emulation_manager,
    health,
    playbooks,
    telemetry,
    wormgpt,
    orchestrator,
    wallet,
    debug
)

router = APIRouter()

# Register all route modules
router.include_router(alerts.router)
router.include_router(cortex.router)
router.include_router(emulation_manager.router)
router.include_router(health.router)
router.include_router(playbooks.router)
router.include_router(telemetry.router)
router.include_router(wormgpt.router)
router.include_router(orchestrator.router)
router.include_router(wallet.router)
router.include_router(debug.router)
