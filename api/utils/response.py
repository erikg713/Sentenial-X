# Sentenial-X/api/utils/response.py

from typing import Any, Dict, Optional
from fastapi.responses import JSONResponse
from fastapi import status

class ResponseBuilder:
    """
    Utility class for building standardized API responses in Sentenial-X.
    Provides consistent formatting for success, error, and exception messages.
    """

    @staticmethod
    def success(
        message: str = "Request successful",
        data: Optional[Dict[str, Any]] = None,
        code: int = status.HTTP_200_OK
    ) -> JSONResponse:
        """
        Return a standardized success response.
        """
        return JSONResponse(
            status_code=code,
            content={
                "status": "success",
                "message": message,
                "data": data or {}
            }
        )

    @staticmethod
    def error(
        message: str = "An error occurred",
        code: int = status.HTTP_400_BAD_REQUEST,
        errors: Optional[Dict[str, Any]] = None
    ) -> JSONResponse:
        """
        Return a standardized error response.
        """
        return JSONResponse(
            status_code=code,
            content={
                "status": "error",
                "message": message,
                "errors": errors or {}
            }
        )

    @staticmethod
    def exception(
        exception: Exception,
        code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        context: Optional[Dict[str, Any]] = None
    ) -> JSONResponse:
        """
        Return a standardized exception response.
        Includes exception details for debugging (hidden in production).
        """
        return JSONResponse(
            status_code=code,
            content={
                "status": "fail",
                "message": str(exception),
                "context": context or {}
            }
        ) 