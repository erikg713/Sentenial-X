"""
Caching Connector Module
========================

Supports Redis or Memcached as a caching layer.
Provides a simple interface for connecting to Redis.
"""

import redis
from redis.exceptions import ConnectionError
from . import logger


def connect_redis(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    password: str | None = None,
    ssl: bool = False,
    socket_timeout: int = 5,
    test_connection: bool = True,
) -> redis.Redis:
    """
    Connect to a Redis cache instance.

    Args:
        host (str): Redis server hostname.
        port (int): Redis server port.
        db (int): Redis database index.
        password (str | None): Redis password if required.
        ssl (bool): Whether to use SSL/TLS connection.
        socket_timeout (int): Timeout for socket connections.
        test_connection (bool): Whether to run a PING on connect.

    Returns:
        redis.Redis: A Redis client instance.
    """
    logger.info(f"Connecting to Redis at {host}:{port}/{db} (SSL={ssl})")

    client = redis.Redis(
        host=host,
        port=port,
        db=db,
        password=password,
        ssl=ssl,
        socket_timeout=socket_timeout,
    )

    if test_connection:
        try:
            client.ping()
            logger.info("Redis connection established successfully ✅")
        except ConnectionError as e:
            logger.error(f"Redis connection failed ❌: {e}")
            raise

    return client
