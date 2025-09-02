# api/utils/auth.py

import os
import jwt
import bcrypt
import logging
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify

logger = logging.getLogger("auth_utils")

# Secret key for signing JWTs
JWT_SECRET = os.getenv("JWT_SECRET", "change_this_secret")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_MINUTES = int(os.getenv("JWT_EXPIRATION_MINUTES", 60))


# -----------------------------
# Password Hashing Utilities
# -----------------------------
def hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))


# -----------------------------
# JWT Token Utilities
# -----------------------------
def generate_token(user_id: str, role: str = "user") -> str:
    """Generate a JWT token for authentication."""
    payload = {
        "user_id": user_id,
        "role": role,
        "exp": datetime.utcnow() + timedelta(minutes=JWT_EXPIRATION_MINUTES),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode a JWT token and return the payload."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        raise ValueError("Token expired")
    except jwt.InvalidTokenError:
        logger.warning("Invalid token")
        raise ValueError("Invalid token")


# -----------------------------
# Flask Decorators
# -----------------------------
def login_required(f):
    """Decorator to ensure the request has a valid JWT token."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        if "Authorization" in request.headers:
            token = request.headers.get("Authorization").split(" ")[-1]

        if not token:
            return jsonify({"error": "Authorization token is missing"}), 401

        try:
            request.user = decode_token(token)
        except ValueError as e:
            return jsonify({"error": str(e)}), 401

        return f(*args, **kwargs)

    return decorated_function


def role_required(required_role):
    """Decorator to enforce role-based access control."""

    def wrapper(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            token = None
            if "Authorization" in request.headers:
                token = request.headers.get("Authorization").split(" ")[-1]

            if not token:
                return jsonify({"error": "Authorization token missing"}), 401

            try:
                payload = decode_token(token)
                if payload.get("role") != required_role:
                    return jsonify({"error": "Unauthorized: insufficient role"}), 403
                request.user = payload
            except ValueError as e:
                return jsonify({"error": str(e)}), 401

            return f(*args, **kwargs)

        return decorated_function

    return wrapper


# -----------------------------
# Pi Network Authentication
# -----------------------------
def validate_pi_token(pi_token: str) -> dict:
    """
    Validate a Pi Network user token.
    This is a stub for real validation with Pi SDK / API.
    """
    if not pi_token or not pi_token.startswith("pi_"):
        logger.warning("Invalid Pi token format")
        raise ValueError("Invalid Pi token")

    # Simulate validation success
    return {
        "user_id": pi_token[3:],  # strip prefix
        "role": "pi_user",
        "validated": True,
    }


# -----------------------------
# Utility for Combined Auth
# -----------------------------
def authenticate_request():
    """
    Authenticate a request via JWT or Pi token.
    Returns user payload or raises ValueError.
    """
    token = None
    if "Authorization" in request.headers:
        token = request.headers.get("Authorization").split(" ")[-1]

    pi_token = request.headers.get("Pi-Token")

    if token:
        return decode_token(token)
    elif pi_token:
        return validate_pi_token(pi_token)
    else:
        raise ValueError("No authentication token provided")
