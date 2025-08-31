# core/semantic_analyzer/models/cache.py

import os
import json
import time
import asyncio
import threading
from collections import OrderedDict
from typing import Any, Optional


class CacheEntry:
    """Represents a single cache entry with value and expiration."""

    def __init__(self, value: Any, ttl: Optional[int] = None):
        self.value = value
        self.timestamp = time.time()
        self.ttl = ttl

    def is_expired(self) -> bool:
        """Check if the entry is expired based on TTL."""
        if self.ttl is None:
            return False
        return (time.time() - self.timestamp) > self.ttl


class Cache:
    """
    A thread-safe cache with optional TTL, LRU eviction, and disk persistence.
    """

    def __init__(self, max_size: int = 1000, persist_file: Optional[str] = None):
        self.max_size = max_size
        self.persist_file = persist_file
        self._lock = threading.Lock()
        self._cache = OrderedDict()

        if self.persist_file and os.path.exists(self.persist_file):
            self._load_from_disk()

    def _evict_if_needed(self):
        """Evict the oldest item if max_size exceeded."""
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set a cache entry with optional TTL."""
        with self._lock:
            self._cache[key] = CacheEntry(value, ttl)
            self._cache.move_to_end(key)  # Mark as most recently used
            self._evict_if_needed()
            self._persist()

    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache if present and not expired."""
        with self._lock:
            entry = self._cache.get(key)
            if not entry:
                return None

            if entry.is_expired():
                del self._cache[key]
                self._persist()
                return None

            # Refresh LRU order
            self._cache.move_to_end(key)
            return entry.value

    def delete(self, key: str):
        """Delete a cache entry if it exists."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._persist()

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._persist()

    def _persist(self):
        """Persist cache to disk if persistence is enabled."""
        if not self.persist_file:
            return
        try:
            serializable_cache = {
                k: {
                    "value": v.value,
                    "timestamp": v.timestamp,
                    "ttl": v.ttl,
                }
                for k, v in self._cache.items()
                if not v.is_expired()
            }
            with open(self.persist_file, "w", encoding="utf-8") as f:
                json.dump(serializable_cache, f)
        except Exception as e:
            print(f"[Cache] Failed to persist cache: {e}")

    def _load_from_disk(self):
        """Load cache from disk if persistence is enabled."""
        try:
            with open(self.persist_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                entry = CacheEntry(v["value"], v["ttl"])
                entry.timestamp = v["timestamp"]
                if not entry.is_expired():
                    self._cache[k] = entry
        except Exception as e:
            print(f"[Cache] Failed to load cache from disk: {e}")

    async def aget(self, key: str) -> Optional[Any]:
        """Async get wrapper."""
        return await asyncio.to_thread(self.get, key)

    async def aset(self, key: str, value: Any, ttl: Optional[int] = None):
        """Async set wrapper."""
        return await asyncio.to_thread(self.set, key, value, ttl)

    async def adelete(self, key: str):
        """Async delete wrapper."""
        return await asyncio.to_thread(self.delete, key)

    async def aclear(self):
        """Async clear wrapper."""
        return await asyncio.to_thread(self.clear)
