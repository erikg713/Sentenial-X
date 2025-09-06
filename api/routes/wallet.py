# -*- coding: utf-8 -*-
"""
Wallet API Routes for Sentenial-X
---------------------------------

Exposes endpoints for wallet management, including balance checks,
funding simulators, and transferring credits between internal accounts.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.simulator.wallet import WalletManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/wallet", tags=["Wallet"])

wallet_manager = WalletManager()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class WalletRequest(BaseModel):
    account_id: str
    amount: float = 0.0
    target_account: str | None = None


class WalletResponse(BaseModel):
    account_id: str
    balance: float
    status: str
    timestamp: datetime


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/balance/{account_id}", response_model=WalletResponse)
async def get_balance(account_id: str) -> WalletResponse:
    """
    Get the current balance of a wallet account.
    """
    try:
        balance = wallet_manager.get_balance(account_id)
        return WalletResponse(
            account_id=account_id,
            balance=balance,
            status="success",
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.exception("Failed to get balance for account %s", account_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fund", response_model=WalletResponse)
async def fund_wallet(request: WalletRequest) -> WalletResponse:
    """
    Add funds to a wallet account.
    """
    try:
        balance = wallet_manager.fund_account(request.account_id, request.amount)
        return WalletResponse(
            account_id=request.account_id,
            balance=balance,
            status="funded",
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.exception("Failed to fund account %s", request.account_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transfer", response_model=WalletResponse)
async def transfer_funds(request: WalletRequest) -> WalletResponse:
    """
    Transfer funds from one wallet account to another.
    """
    if not request.target_account:
        raise HTTPException(status_code=400, detail="Target account is required for transfer")
    try:
        balance = wallet_manager.transfer(
            from_account=request.account_id,
            to_account=request.target_account,
            amount=request.amount
        )
        return WalletResponse(
            account_id=request.account_id,
            balance=balance,
            status=f"transferred {request.amount} to {request.target_account}",
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.exception("Failed to transfer from %s to %s", request.account_id, request.target_account)
        raise HTTPException(status_code=500, detail=str(e))
