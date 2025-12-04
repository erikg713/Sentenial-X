# -*- coding: utf-8 -*-
"""
JWT auth adapter
----------------

Small, focused adapter to create and verify JWTs. The implementation aims to be:
- readable and idiomatic,
- small surface area (encode / decode / verify helpers),
- easily testable and extensible (injectable key resolver for JWKS / multi-key setups).

This file intentionally avoids network calls by default — if you need JWKS fetching,
pass a `key_resolver` callable that knows how to fetch keys (e.g. from a JWKS endpoint).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Union

import jwt  # PyJWT

LOGGER = logging.getLogger(__name__)

# Public types
JSON = Mapping[str, Any]
KeyResolver = Callable[[str], str]  # given a 'kid', return PEM/public key string


class AuthError(Exception):
    """Generic authentication error raised by this adapter."""


class TokenExpired(AuthError):
    """Token has expired."""


class InvalidToken(AuthError):
    """Token is invalid or could not be decoded."""


@dataclass(frozen=True)
class JwtConfig:
    """
    Configuration object for the adapter.

    Attributes:
    - algorithm: default signing algorithm (HS256, RS256, etc.)
    - leeway: seconds of clock skew to allow when verifying exp/nbf
    - issuer: expected issuer claim (optional)
    - audience: expected audience claim (optional)
    """
    algorithm: str = "HS256"
    leeway: int = 0
    issuer: Optional[str] = None
    audience: Optional[Union[str, Iterable[str]]] = None


class JwtAuthAdapter:
    """
    JWT helper for creating and validating tokens.

    Design notes:
    - Keep logic framework-agnostic: this class returns dicts and raises exceptions.
    - For RS256 / multi-key setups, provide a `key_resolver` that accepts a `kid`
      value and returns a PEM-formatted key. If no resolver is provided, `key`
      must be a raw secret/public-key string passed to encode/decode.
    """

    def __init__(self, config: Optional[JwtConfig] = None, key_resolver: Optional[KeyResolver] = None):
        self._config = config or JwtConfig()
        self._key_resolver = key_resolver

    def encode(
        self,
        payload: Dict[str, Any],
        key: str,
        *,
        expires_in: Optional[Union[int, timedelta]] = None,
        algorithm: Optional[str] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a signed JWT.

        - payload: claim set (will be shallow-copied and "iat"/"exp" added if missing).
        - key: secret (for HS*) or private key PEM (for RS*/ES*).
        - expires_in: seconds or timedelta; if None, no exp is set.
        - algorithm: override default algorithm from config.
        - headers: optional JOSE headers (e.g. {"kid": "abc"}).

        Returns the compact JWT string.
        """
        if algorithm is None:
            algorithm = self._config.algorithm

        claims = dict(payload)  # avoid mutating caller's payload
        now = datetime.now(timezone.utc)
        if "iat" not in claims:
            claims["iat"] = int(now.timestamp())

        if expires_in is not None:
            if isinstance(expires_in, timedelta):
                exp_ts = now + expires_in
            else:
                exp_ts = now + timedelta(seconds=int(expires_in))
            claims["exp"] = int(exp_ts.timestamp())

        LOGGER.debug("Encoding JWT (alg=%s) with payload keys=%s", algorithm, sorted(claims.keys()))

        token = jwt.encode(claims, key, algorithm=algorithm, headers=headers)
        # PyJWT returns bytes for some algorithm/key combos under older versions; ensure str
        if isinstance(token, bytes):
            token = token.decode("utf-8")
        return token

    def decode(
        self,
        token: str,
        key: Optional[str] = None,
        *,
        algorithms: Optional[Iterable[str]] = None,
        verify_exp: bool = True,
        audience: Optional[Union[str, Iterable[str]]] = None,
        issuer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Decode and validate a JWT.

        - token: compact JWT string.
        - key: secret/public key to verify signature. If omitted and a key_resolver
               is configured, adapter will attempt to extract `kid` and resolve the key.
        - algorithms: list of accepted algorithms (defaults to adapter config.algorithm).
        - verify_exp: whether to check expiration (exp claim).
        - audience / issuer: override adapter config expectations.

        Returns the decoded claims as a dict. Raises TokenExpired or InvalidToken.
        """
        algorithms = list(algorithms) if algorithms is not None else [self._config.algorithm]
        audience = audience if audience is not None else self._config.audience
        issuer = issuer if issuer is not None else self._config.issuer

        # Early quick validation
        if key is None and self._key_resolver is None:
            raise InvalidToken("No verification key provided and no key_resolver configured.")

        # If key is not provided, try to resolve from token header 'kid'
        if key is None:
            kid = self._get_kid_from_token(token)
            if not kid:
                raise InvalidToken("Token missing 'kid' header and no key provided.")
            key = self._key_resolver(kid)
            if not key:
                raise InvalidToken(f"Key resolver did not return a key for kid={kid}")

        options = {"verify_exp": bool(verify_exp)}
        try:
            claims = jwt.decode(
                token,
                key,
                algorithms=algorithms,
                audience=audience,
                issuer=issuer,
                leeway=self._config.leeway,
                options=options,
            )
            LOGGER.debug("Decoded JWT successfully; claims keys=%s", sorted(claims.keys()))
            return claims
        except jwt.ExpiredSignatureError as exc:
            LOGGER.info("Expired JWT: %s", exc)
            raise TokenExpired("Token has expired") from exc
        except jwt.InvalidTokenError as exc:
            LOGGER.warning("Invalid JWT: %s", exc)
            raise InvalidToken(f"Invalid token: {exc}") from exc

    @staticmethod
    def _get_kid_from_token(token: str) -> Optional[str]:
        """
        Extract the 'kid' value from the JWT header without verifying the signature.
        Returns None if the header cannot be decoded or no kid is present.
        """
        try:
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header.get("kid")
            LOGGER.debug("Extracted kid=%s from token header", kid)
            return kid
        except jwt.DecodeError:
            LOGGER.debug("Failed to decode token header to extract kid")
            return None
        except Exception:  # pragma: no cover - defensive
            LOGGER.exception("Unexpected error extracting kid from token header")
            return None
