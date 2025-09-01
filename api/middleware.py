# api/middleware.py
"""
Middleware for Sentenial-X API
Includes API key verification, logging, and error handling
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from api.utils.logger import init_logger
from api.utils.response import error_response
from api.config import API_KEY

logger = init_logger("sentenialx_middleware")


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Middleware to verify X-API-KEY header in requests
    """

    async def dispatch(self, request: Request, call_next):
        api_key = request.headers.get("x-api-key")
        if api_key != API_KEY:
            logger.warning(f"Unauthorized access attempt from {request.client.host}")
            return JSONResponse(
                status_code=403,
                content=error_response("Invalid API Key", 403)
            )
        response = await call_next(request)
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log incoming requests
    """

    async def dispatch(self, request: Request, call_next):
        logger.info(f"Incoming request: {request.method} {request.url}")
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response


class GlobalExceptionMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle uncaught exceptions
    """

    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except HTTPException as e:
            logger.error(f"HTTPException: {e.detail}")
            return JSONResponse(
                status_code=e.status_code,
                content=error_response(e.detail, e.status_code)
            )
        except Exception as e:
            logger.exception("Unhandled exception occurred")
            return JSONResponse(
                status_code=500,
                content=error_response("Internal Server Error", 500)
            )
