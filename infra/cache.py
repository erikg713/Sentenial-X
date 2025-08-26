"""
Caching Connector Module
========================

Supports Redis or Memcached as a caching layer.
"""

import redis
from . import logger


def connect_redis(host="localhost", port=6379, db=0):
    """Connect to Redis cache."""
    logger.info(f"Connecting to Redis at {host}:{port}/{db}")
    return redis.Redis(host=host, port=port, db=db)
