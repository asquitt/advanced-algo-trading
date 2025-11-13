"""
Redis cache implementation to reduce API costs.

Caching strategy:
- Market data: Cache for 1-5 minutes (depends on volatility)
- LLM analysis: Cache for 24 hours (expensive to regenerate)
- News/filings: Cache for 1 hour
- Quotes: Cache for 15 seconds during market hours

This can save hundreds of dollars in API costs per month!
"""

import redis
import json
from typing import Optional, Any
from datetime import timedelta
from src.utils.config import settings
from src.utils.logger import app_logger
from functools import wraps


class Cache:
    """
    Redis-based cache with helper methods for common operations.
    """

    def __init__(self):
        """Initialize Redis connection with automatic retry."""
        try:
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True,
                health_check_interval=30,
            )
            # Test connection
            self.redis_client.ping()
            app_logger.info("Redis cache connected successfully")
        except redis.ConnectionError as e:
            app_logger.error(f"Redis connection failed: {e}")
            self.redis_client = None

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value (automatically deserialized from JSON) or None
        """
        if not self.redis_client:
            return None

        try:
            value = self.redis_client.get(key)
            if value:
                app_logger.debug(f"Cache HIT: {key}")
                return json.loads(value)
            else:
                app_logger.debug(f"Cache MISS: {key}")
                return None
        except Exception as e:
            app_logger.error(f"Cache get error for {key}: {e}")
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache with optional TTL.

        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl: Time to live in seconds (None = no expiration)

        Returns:
            True if successful, False otherwise
        """
        if not self.redis_client:
            return False

        try:
            serialized = json.dumps(value)
            if ttl:
                self.redis_client.setex(key, ttl, serialized)
            else:
                self.redis_client.set(key, serialized)
            app_logger.debug(f"Cache SET: {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            app_logger.error(f"Cache set error for {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self.redis_client:
            return False

        try:
            self.redis_client.delete(key)
            app_logger.debug(f"Cache DELETE: {key}")
            return True
        except Exception as e:
            app_logger.error(f"Cache delete error for {key}: {e}")
            return False

    def clear_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern.

        Args:
            pattern: Redis pattern (e.g., "llm_analysis:*")

        Returns:
            Number of keys deleted
        """
        if not self.redis_client:
            return 0

        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                count = self.redis_client.delete(*keys)
                app_logger.info(f"Cleared {count} keys matching {pattern}")
                return count
            return 0
        except Exception as e:
            app_logger.error(f"Cache clear pattern error for {pattern}: {e}")
            return 0


def cached(ttl: int = 3600, key_prefix: str = ""):
    """
    Decorator for caching function results.

    Usage:
        @cached(ttl=3600, key_prefix="market_data")
        def get_stock_quote(symbol: str):
            # Expensive API call here
            return fetch_from_api(symbol)

    Args:
        ttl: Cache time-to-live in seconds
        key_prefix: Prefix for cache key
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_key = f"{key_prefix}:{func.__name__}:{str(args)}:{str(kwargs)}"

            # Try to get from cache
            cache = Cache()
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl=ttl)
            return result

        return wrapper
    return decorator


# Global cache instance
cache = Cache()
