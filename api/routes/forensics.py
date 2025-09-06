# -*- coding: utf-8 -*-
"""
Forensics API Routes
--------------------

Provides forensic log access and analysis.
"""

from __future__ import annotations
from fastapi import APIRouter
from datetime import datetime

router = APIRouter(prefix="/forensics", tags=["Forensics"])


@router.get("/logs")
async def forensic_logs() -> list[dict]:
    return [
        {"id": "F001", "event": "File modification", "timestamp": datetime.utcnow().isoformat()},
        {"id": "F002", "event": "Registry tampering", "timestamp": datetime.utcnow().isoformat()},
    ]
