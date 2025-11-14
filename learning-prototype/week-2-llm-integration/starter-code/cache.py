"""
Redis Caching for LLM Responses

This module implements Redis-based caching to reduce API costs and latency.
Caching LLM responses can reduce costs by 90%+ for repeated queries.

Learning Goals:
- Understand caching fundamentals
- Work with Redis
- Implement decorators
- Design TTL strategies
- Handle cache invalidation

Cost Impact:
- Without cache: $10/week
- With cache (1 hour TTL): $1/week
- With cache (1 day TTL): $0.10/week
"""

import os
import json
import hashlib
import functools
from typing import Any, Optional, Callable, Dict, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio

# TODO: Install Redis
# pip install redis[hiredis]


# =============================================================================
# TODO #1: Setup Redis connection
# =============================================================================
# Instructions:
# 1. Import redis library
# 2. Create RedisCache class
# 3. Implement connection with retry logic
# 4. Add health check method
# 5. Handle connection failures gracefully
#
# Redis setup:
# - Local: docker run -d -p 6379:6379 redis
# - Cloud: Use Redis Cloud, AWS ElastiCache, etc.
# - Environment variable: REDIS_URL
# =============================================================================


@dataclass
class CacheConfig:
    """Configuration for Redis cache"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    default_ttl: int = 3600  # 1 hour in seconds
    max_retries: int = 3
    socket_timeout: int = 5


class RedisCache:
    """
    Redis-based cache for LLM responses.

    Provides get/set operations with TTL support and automatic
    serialization/deserialization of Python objects.

    Example:
        cache = RedisCache()
        cache.set("key", {"data": "value"}, ttl=3600)
        value = cache.get("key")
    """

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        redis_url: Optional[str] = None,
        debug: bool = False
    ):
        """
        Initialize Redis cache.

        TODO: Implement initialization
        Steps:
        1. Load config from parameter or create default
        2. Get redis_url from parameter or environment
        3. Import redis library
        4. Create Redis client (handle both URL and individual params)
        5. Test connection with ping()
        6. Set debug flag

        Args:
            config: Cache configuration
            redis_url: Redis connection URL (overrides config)
            debug: Enable debug logging

        Raises:
            ConnectionError: If unable to connect to Redis
        """
        # TODO: Implement Redis connection
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # import redis
        #
        # self.config = config or CacheConfig()
        # self.debug = debug
        # self.stats = {
        #     "hits": 0,
        #     "misses": 0,
        #     "sets": 0,
        #     "deletes": 0,
        #     "errors": 0
        # }
        #
        # # Get Redis URL from parameter or environment
        # redis_url = redis_url or os.getenv("REDIS_URL")
        #
        # try:
        #     if redis_url:
        #         self.client = redis.from_url(
        #             redis_url,
        #             decode_responses=True,
        #             socket_timeout=self.config.socket_timeout
        #         )
        #     else:
        #         self.client = redis.Redis(
        #             host=self.config.host,
        #             port=self.config.port,
        #             db=self.config.db,
        #             password=self.config.password,
        #             decode_responses=True,
        #             socket_timeout=self.config.socket_timeout
        #         )
        #
        #     # Test connection
        #     self.client.ping()
        #     self._log("Redis connection established")
        #
        # except Exception as e:
        #     self._log(f"Redis connection failed: {str(e)}", "ERROR")
        #     raise ConnectionError(f"Could not connect to Redis: {str(e)}")

    def _log(self, message: str, level: str = "INFO"):
        """Log a message if debug is enabled"""
        if self.debug:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [CACHE] [{level}] {message}")

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.

        TODO: Implement cache get
        Steps:
        1. Try to get value from Redis
        2. If found, deserialize JSON and return
        3. If not found, return None
        4. Update stats (hit or miss)
        5. Log the operation
        6. Handle errors gracefully

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        # TODO: Implement cache get
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # try:
        #     value = self.client.get(key)
        #     if value is not None:
        #         self.stats["hits"] += 1
        #         self._log(f"Cache HIT: {key}")
        #         return json.loads(value)
        #     else:
        #         self.stats["misses"] += 1
        #         self._log(f"Cache MISS: {key}")
        #         return None
        # except Exception as e:
        #     self.stats["errors"] += 1
        #     self._log(f"Cache get error: {str(e)}", "ERROR")
        #     return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set a value in cache.

        TODO: Implement cache set
        Steps:
        1. Serialize value to JSON
        2. Set in Redis with TTL
        3. Update stats
        4. Log the operation
        5. Handle errors gracefully
        6. Return True on success, False on failure

        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Time to live in seconds (None = use default)

        Returns:
            True if successful, False otherwise
        """
        # TODO: Implement cache set
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # try:
        #     ttl = ttl or self.config.default_ttl
        #     serialized = json.dumps(value)
        #     self.client.setex(key, ttl, serialized)
        #     self.stats["sets"] += 1
        #     self._log(f"Cache SET: {key} (TTL: {ttl}s)")
        #     return True
        # except Exception as e:
        #     self.stats["errors"] += 1
        #     self._log(f"Cache set error: {str(e)}", "ERROR")
        #     return False

    def delete(self, key: str) -> bool:
        """
        Delete a key from cache.

        TODO: Implement cache delete
        Steps:
        1. Delete key from Redis
        2. Update stats
        3. Log the operation
        4. Return success status

        Args:
            key: Cache key to delete

        Returns:
            True if deleted, False otherwise
        """
        # TODO: Implement cache delete
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # try:
        #     deleted = self.client.delete(key)
        #     if deleted:
        #         self.stats["deletes"] += 1
        #         self._log(f"Cache DELETE: {key}")
        #     return bool(deleted)
        # except Exception as e:
        #     self.stats["errors"] += 1
        #     self._log(f"Cache delete error: {str(e)}", "ERROR")
        #     return False

    def exists(self, key: str) -> bool:
        """
        Check if a key exists in cache.

        TODO: Implement existence check
        - Use Redis EXISTS command
        - Return boolean

        Args:
            key: Cache key

        Returns:
            True if key exists, False otherwise
        """
        # TODO: Implement exists check
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # try:
        #     return bool(self.client.exists(key))
        # except Exception as e:
        #     self._log(f"Cache exists error: {str(e)}", "ERROR")
        #     return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats including hit rate
        """
        stats = self.stats.copy()
        total_requests = stats["hits"] + stats["misses"]
        if total_requests > 0:
            stats["hit_rate"] = stats["hits"] / total_requests
        else:
            stats["hit_rate"] = 0.0
        return stats


# =============================================================================
# TODO #2: Implement cache decorator
# =============================================================================
# Instructions:
# 1. Create @cache_response decorator
# 2. Generate cache key from function args
# 3. Check cache before calling function
# 4. Cache result after function call
# 5. Support both sync and async functions
#
# This decorator makes caching transparent:
# @cache_response(ttl=3600)
# async def expensive_function(arg1, arg2):
#     ...
# =============================================================================


def generate_cache_key(
    prefix: str,
    *args,
    **kwargs
) -> str:
    """
    Generate a cache key from function arguments.

    TODO: Implement cache key generation
    Steps:
    1. Combine prefix, args, and kwargs into a string
    2. Hash the string (MD5 is fast enough)
    3. Return formatted key: "prefix:hash"
    4. Ensure consistent ordering of kwargs

    Args:
        prefix: Key prefix (usually function name)
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Cache key string
    """
    # TODO: Implement cache key generation
    pass

    # EXAMPLE IMPLEMENTATION (uncomment to use):
    # # Create a stable string representation
    # key_parts = [prefix]
    #
    # # Add args
    # for arg in args:
    #     if hasattr(arg, '__dict__'):
    #         key_parts.append(str(sorted(arg.__dict__.items())))
    #     else:
    #         key_parts.append(str(arg))
    #
    # # Add kwargs (sorted for consistency)
    # for k, v in sorted(kwargs.items()):
    #     key_parts.append(f"{k}={v}")
    #
    # # Create hash
    # key_str = ":".join(key_parts)
    # hash_digest = hashlib.md5(key_str.encode()).hexdigest()[:12]
    #
    # return f"{prefix}:{hash_digest}"


def cache_response(
    ttl: int = 3600,
    cache_instance: Optional[RedisCache] = None,
    key_prefix: Optional[str] = None
):
    """
    Decorator to cache function responses.

    TODO: Implement cache decorator
    Steps:
    1. Create decorator that wraps function
    2. Generate cache key from function name and args
    3. Check cache before calling function
    4. If cache hit, return cached value
    5. If cache miss, call function and cache result
    6. Handle both sync and async functions
    7. Preserve function metadata with functools.wraps

    Args:
        ttl: Time to live in seconds
        cache_instance: Redis cache instance (creates new if None)
        key_prefix: Custom key prefix (uses function name if None)

    Returns:
        Decorated function

    Example:
        @cache_response(ttl=3600)
        async def analyze_stock(symbol, metrics):
            # Expensive LLM call
            return result
    """
    # TODO: Implement decorator
    pass

    # EXAMPLE IMPLEMENTATION (uncomment to use):
    # def decorator(func: Callable) -> Callable:
    #     cache = cache_instance or RedisCache()
    #     prefix = key_prefix or func.__name__
    #
    #     @functools.wraps(func)
    #     async def async_wrapper(*args, **kwargs):
    #         # Generate cache key
    #         cache_key = generate_cache_key(prefix, *args, **kwargs)
    #
    #         # Try cache
    #         cached = cache.get(cache_key)
    #         if cached is not None:
    #             return cached
    #
    #         # Cache miss - call function
    #         result = await func(*args, **kwargs)
    #
    #         # Cache result
    #         cache.set(cache_key, result, ttl=ttl)
    #
    #         return result
    #
    #     @functools.wraps(func)
    #     def sync_wrapper(*args, **kwargs):
    #         cache_key = generate_cache_key(prefix, *args, **kwargs)
    #
    #         cached = cache.get(cache_key)
    #         if cached is not None:
    #             return cached
    #
    #         result = func(*args, **kwargs)
    #         cache.set(cache_key, result, ttl=ttl)
    #
    #         return result
    #
    #     # Return appropriate wrapper based on function type
    #     if asyncio.iscoroutinefunction(func):
    #         return async_wrapper
    #     else:
    #         return sync_wrapper
    #
    # return decorator


# =============================================================================
# TODO #3: Add cache invalidation
# =============================================================================
# Instructions:
# 1. Create invalidate_pattern() method to clear keys by pattern
# 2. Add invalidate_stock() to clear all data for a symbol
# 3. Implement time-based invalidation
# 4. Add manual invalidation triggers
# 5. Create scheduled invalidation jobs
#
# Invalidation strategies:
# - On data update: Clear specific symbol
# - On market close: Clear all intraday data
# - On earnings: Clear company-specific analyses
# - On error: Clear potentially stale data
# =============================================================================


class RedisCache:  # Continued from above
    """Additional methods for cache invalidation"""

    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching a pattern.

        TODO: Implement pattern-based invalidation
        Steps:
        1. Use Redis SCAN to find matching keys
        2. Delete each key
        3. Count deleted keys
        4. Log the operation
        5. Return count

        Args:
            pattern: Redis key pattern (e.g., "analysis:AAPL:*")

        Returns:
            Number of keys deleted

        Warning:
            SCAN is safer than KEYS for production use
        """
        # TODO: Implement pattern invalidation
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # try:
        #     deleted = 0
        #     cursor = 0
        #
        #     while True:
        #         cursor, keys = self.client.scan(
        #             cursor=cursor,
        #             match=pattern,
        #             count=100
        #         )
        #
        #         if keys:
        #             deleted += self.client.delete(*keys)
        #
        #         if cursor == 0:
        #             break
        #
        #     self._log(f"Invalidated {deleted} keys matching {pattern}")
        #     return deleted
        #
        # except Exception as e:
        #     self._log(f"Invalidation error: {str(e)}", "ERROR")
        #     return 0

    def invalidate_stock(self, symbol: str) -> int:
        """
        Invalidate all cached data for a stock symbol.

        TODO: Implement stock-specific invalidation
        - Call invalidate_pattern with symbol-based pattern
        - Return count of invalidated keys

        Args:
            symbol: Stock symbol

        Returns:
            Number of keys deleted
        """
        # TODO: Implement stock invalidation
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # pattern = f"*:{symbol}:*"
        # return self.invalidate_pattern(pattern)

    def clear_expired(self) -> int:
        """
        Manually trigger clearing of expired keys.

        Note: Redis handles this automatically, but you can
        force it for testing or when using Redis as LRU cache.

        TODO: Implement expired key cleanup
        - Find keys with TTL <= 0
        - Delete them
        - Return count

        Returns:
            Number of expired keys deleted
        """
        # TODO: Implement expired cleanup
        pass

    def clear_all(self) -> bool:
        """
        Clear all keys in the cache.

        WARNING: Use with caution! This clears EVERYTHING.

        TODO: Implement full cache clear
        - Use FLUSHDB command
        - Log the operation
        - Return success status

        Returns:
            True if successful
        """
        # TODO: Implement full clear
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # try:
        #     self.client.flushdb()
        #     self._log("Cache cleared completely", "WARNING")
        #     return True
        # except Exception as e:
        #     self._log(f"Clear all error: {str(e)}", "ERROR")
        #     return False


# =============================================================================
# TODO #4: Add TTL strategies
# =============================================================================
# Instructions:
# 1. Create TTL strategy class
# 2. Implement different strategies (fixed, dynamic, adaptive)
# 3. Add volatility-based TTL
# 4. Implement time-of-day adjustments
# 5. Create strategy factory
#
# TTL Strategies:
# - Fixed: Same TTL for all (simple)
# - Dynamic: Based on volatility (shorter for volatile stocks)
# - Adaptive: Learn from cache hit patterns
# - Time-based: Shorter during market hours
# =============================================================================


class TTLStrategy:
    """Base class for TTL calculation strategies"""

    def calculate_ttl(
        self,
        symbol: str,
        base_ttl: int,
        context: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Calculate TTL for a cache entry.

        TODO: Implement in subclasses
        - Take symbol and context
        - Return appropriate TTL in seconds

        Args:
            symbol: Stock symbol
            base_ttl: Base TTL in seconds
            context: Additional context (volatility, time, etc.)

        Returns:
            Calculated TTL in seconds
        """
        return base_ttl


class FixedTTLStrategy(TTLStrategy):
    """Simple fixed TTL strategy"""

    def calculate_ttl(
        self,
        symbol: str,
        base_ttl: int,
        context: Optional[Dict[str, Any]] = None
    ) -> int:
        """Always return base TTL"""
        return base_ttl


class VolatilityBasedTTLStrategy(TTLStrategy):
    """
    Adjust TTL based on stock volatility.

    High volatility = shorter TTL (data gets stale faster)
    Low volatility = longer TTL (data stays relevant longer)
    """

    def calculate_ttl(
        self,
        symbol: str,
        base_ttl: int,
        context: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Calculate TTL based on volatility.

        TODO: Implement volatility-based TTL
        Steps:
        1. Get volatility from context (e.g., beta value)
        2. High volatility (beta > 1.5): TTL = base_ttl * 0.5
        3. Low volatility (beta < 0.5): TTL = base_ttl * 2.0
        4. Normal volatility: TTL = base_ttl
        5. Enforce min/max bounds

        Args:
            symbol: Stock symbol
            base_ttl: Base TTL
            context: Dict with 'volatility' or 'beta' key

        Returns:
            Adjusted TTL
        """
        # TODO: Implement volatility-based calculation
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # if not context:
        #     return base_ttl
        #
        # # Get volatility measure
        # beta = context.get('beta', 1.0)
        #
        # # Adjust TTL based on volatility
        # if beta > 1.5:
        #     # High volatility - shorter TTL
        #     ttl = int(base_ttl * 0.5)
        # elif beta < 0.5:
        #     # Low volatility - longer TTL
        #     ttl = int(base_ttl * 2.0)
        # else:
        #     # Normal volatility
        #     ttl = base_ttl
        #
        # # Enforce bounds (min 5 minutes, max 24 hours)
        # return max(300, min(ttl, 86400))


class TimeBasedTTLStrategy(TTLStrategy):
    """
    Adjust TTL based on time of day.

    During market hours: shorter TTL
    After market close: longer TTL
    """

    def calculate_ttl(
        self,
        symbol: str,
        base_ttl: int,
        context: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Calculate TTL based on time of day.

        TODO: Implement time-based TTL
        Steps:
        1. Get current time
        2. Check if market is open (9:30 AM - 4:00 PM ET)
        3. Market open: TTL = base_ttl * 0.5
        4. Market closed: TTL = base_ttl * 2.0
        5. Weekend: TTL = base_ttl * 3.0

        Args:
            symbol: Stock symbol
            base_ttl: Base TTL
            context: Optional context

        Returns:
            Adjusted TTL
        """
        # TODO: Implement time-based calculation
        pass

        # EXAMPLE IMPLEMENTATION (uncomment to use):
        # from datetime import datetime
        # import pytz
        #
        # now = datetime.now(pytz.timezone('America/New_York'))
        # hour = now.hour
        # weekday = now.weekday()
        #
        # # Weekend
        # if weekday >= 5:
        #     return int(base_ttl * 3.0)
        #
        # # Market hours (9:30 AM - 4:00 PM ET)
        # if 9 <= hour < 16:
        #     return int(base_ttl * 0.5)
        #
        # # After hours
        # return int(base_ttl * 2.0)


def create_ttl_strategy(strategy_type: str = "fixed") -> TTLStrategy:
    """
    Factory function for TTL strategies.

    TODO: Implement strategy factory
    - Map strategy name to class
    - Return instance

    Args:
        strategy_type: Type of strategy (fixed, volatility, time)

    Returns:
        TTLStrategy instance
    """
    # TODO: Implement strategy factory
    pass

    # EXAMPLE IMPLEMENTATION (uncomment to use):
    # strategies = {
    #     "fixed": FixedTTLStrategy,
    #     "volatility": VolatilityBasedTTLStrategy,
    #     "time": TimeBasedTTLStrategy,
    # }
    #
    # strategy_class = strategies.get(strategy_type, FixedTTLStrategy)
    # return strategy_class()


# =============================================================================
# Example Usage
# =============================================================================

async def example_usage():
    """
    Example of how to use the caching system.

    Run this after implementing all TODOs to test your code.
    """
    print("=== Initializing Redis Cache ===")
    cache = RedisCache(debug=True)

    # Test basic operations
    print("\n=== Testing Basic Operations ===")
    cache.set("test_key", {"data": "test_value"}, ttl=60)
    value = cache.get("test_key")
    print(f"Retrieved: {value}")

    # Test decorator
    print("\n=== Testing Cache Decorator ===")

    @cache_response(ttl=300)
    async def expensive_operation(symbol: str) -> Dict[str, Any]:
        """Simulate expensive LLM call"""
        print(f"  -> Calling expensive operation for {symbol}")
        await asyncio.sleep(0.5)  # Simulate API delay
        return {
            "symbol": symbol,
            "recommendation": "BUY",
            "confidence": 0.85
        }

    # First call - should be slow
    print("First call (cache miss):")
    result1 = await expensive_operation("AAPL")
    print(f"Result: {result1}")

    # Second call - should be fast (cached)
    print("\nSecond call (cache hit):")
    result2 = await expensive_operation("AAPL")
    print(f"Result: {result2}")

    # Test TTL strategies
    print("\n=== Testing TTL Strategies ===")
    volatility_strategy = create_ttl_strategy("volatility")
    base_ttl = 3600

    # High volatility stock
    ttl_high_vol = volatility_strategy.calculate_ttl(
        "TSLA",
        base_ttl,
        context={"beta": 2.0}
    )
    print(f"High volatility TTL: {ttl_high_vol}s ({ttl_high_vol / 60:.1f} min)")

    # Low volatility stock
    ttl_low_vol = volatility_strategy.calculate_ttl(
        "KO",
        base_ttl,
        context={"beta": 0.4}
    )
    print(f"Low volatility TTL: {ttl_low_vol}s ({ttl_low_vol / 60:.1f} min)")

    # Test invalidation
    print("\n=== Testing Invalidation ===")
    cache.set("analysis:AAPL:abc123", {"data": "test"}, ttl=3600)
    cache.set("analysis:AAPL:def456", {"data": "test2"}, ttl=3600)
    cache.set("analysis:TSLA:xyz789", {"data": "test3"}, ttl=3600)

    deleted = cache.invalidate_stock("AAPL")
    print(f"Invalidated {deleted} keys for AAPL")

    # Show stats
    print("\n=== Cache Statistics ===")
    stats = cache.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


def test_cache_savings():
    """
    Calculate cost savings from caching.

    Shows the dramatic impact of caching on API costs.
    """
    print("=== Cache Cost Impact Analysis ===\n")

    # Assumptions
    requests_per_day = 1000
    cost_per_request = 0.001  # $0.001 per request
    cache_hit_rates = [0.0, 0.5, 0.75, 0.90, 0.95]

    for hit_rate in cache_hit_rates:
        uncached_requests = requests_per_day * (1 - hit_rate)
        daily_cost = uncached_requests * cost_per_request
        monthly_cost = daily_cost * 30

        print(f"Cache Hit Rate: {hit_rate * 100:.0f}%")
        print(f"  Requests to LLM: {uncached_requests:.0f}/day")
        print(f"  Daily cost: ${daily_cost:.2f}")
        print(f"  Monthly cost: ${monthly_cost:.2f}")
        print(f"  Savings: ${(requests_per_day * cost_per_request - daily_cost) * 30:.2f}/month")
        print()


if __name__ == "__main__":
    # Uncomment to test your implementation
    # asyncio.run(example_usage())

    # Show cost impact
    # test_cache_savings()
    pass
