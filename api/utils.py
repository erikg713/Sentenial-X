# -*- coding: utf-8 -*-
"""
Sentenial-X API Utilities
-------------------------

Provides helper functions for:
- Standardized API responses
- Error handling
- Validation
- Logging helpers
"""

from __future__ import annotations

import logging
from fastapi import HTTPException
from typing import Any, Dict, Optional, Callable
from functools import wraps

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("SentenialX.API.Utils")
if not logger.handlers:
    # Default to NullHandler to avoid logging if not configured
    logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Standard API response
# ---------------------------------------------------------------------------
def api_response(data: Any = None, success: bool = True, message: Optional[str] = None) -> Dict[str, Any]:
    """
    Standard JSON response format for the API
    """
    response: Dict[str, Any] = {"success": success, "data": data}
    if message:
        response["message"] = message
    return response

# ---------------------------------------------------------------------------
# Exception helper
# ---------------------------------------------------------------------------
def api_exception(status_code: int, detail: str) -> None:
    """
    Raise a standardized HTTPException
    """
    logger.warning("API Exception raised: %s (status %s)", detail, status_code)
    raise HTTPException(status_code=status_code, detail=detail)

# ---------------------------------------------------------------------------
# Validation decorators
# ---------------------------------------------------------------------------
def validate_not_empty(param_name: str) -> Callable:
    """
    Decorator to ensure a function argument is not empty or None
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            value = kwargs.get(param_name) or getattr(args[0], param_name, None)
            if value is None or (hasattr(value, "__len__") and len(value) == 0):
                api_exception(400, f"Parameter '{param_name}' cannot be empty")
            return func(*args, **kwargs)
        return wrapper
    return decorator

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def log_api_call(func: Callable) -> Callable:
    """
    Decorator to automatically log API calls
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info("API call: %s args=%s kwargs=%s", func.__name__, args, kwargs)
        result = func(*args, **kwargs)
        logger.info("API call result: %s", result)
        return result
    return wrapper

# ---------------------------------------------------------------------------
# Example usage (for internal testing)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    @log_api_call
    @validate_not_empty("param")
    def test_api(param):
        return api_response({"param": param}, message="Test successful")

    print(test_api(param="hello"))
    try:
        print(test_api(param=""))
    except HTTPException as e:
        print(f"Caught exception: {e.detail}")
