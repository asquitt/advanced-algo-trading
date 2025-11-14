# Caching Strategies for LLM-Powered Trading

**Time to Read**: 20-25 minutes | **Difficulty**: Intermediate

---

## Table of Contents

1. [Why Cache LLM Responses?](#why-cache-llm-responses)
2. [Redis Setup](#redis-setup)
3. [Cache Key Design](#cache-key-design)
4. [TTL Strategies](#ttl-strategies)
5. [Cache Invalidation](#cache-invalidation)
6. [Cost Savings Analysis](#cost-savings-analysis)
7. [Monitoring and Debugging](#monitoring-and-debugging)

---

## Why Cache LLM Responses?

Caching LLM responses can reduce costs by **90%+** and improve latency by **10-100x**. Here's why it's essential for trading systems.

### The Problem: Cost and Latency

**Without Caching**:
```python
# Every request hits the LLM
for symbol in ["AAPL", "MSFT", "GOOGL"]:
    analysis = await llm.analyze(symbol, data)  # $0.01 each, 500ms each
    # Total: $0.03, 1500ms
```

**With Caching**:
```python
# First request: LLM
analysis = await cached_analyze("AAPL", data)  # $0.01, 500ms

# Second request (within 1 hour): Cache
analysis = await cached_analyze("AAPL", data)  # $0, 5ms

# 90% cost reduction, 100x speedup!
```

### Real Trading Scenario

You're running a trading bot that:
- Analyzes 50 stocks every 15 minutes
- Markets open 6.5 hours/day (390 minutes)
- That's 26 analysis cycles per day
- 50 stocks × 26 cycles = **1,300 analyses/day**

**Without Cache**:
- 1,300 × $0.01 = **$13/day = $390/month**
- Average latency: 500ms per analysis

**With Cache (1-hour TTL)**:
- Stock data changes infrequently
- 90% cache hit rate
- 1,300 × 0.1 × $0.01 = **$1.30/day = $39/month**
- Average latency: 50ms (mostly from cache)

**Savings: $351/month + 10x faster responses**

---

## Redis Setup

Redis is the gold standard for caching. It's fast, reliable, and easy to use.

### Installation

**Using Docker** (recommended):
```bash
docker run -d \
  --name redis-cache \
  -p 6379:6379 \
  redis:7-alpine \
  redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
```

**Or install locally**:
```bash
# macOS
brew install redis
redis-server

# Ubuntu
sudo apt-get install redis-server
sudo systemctl start redis
```

### Python Client Setup

```python
import redis.asyncio as redis
import json
from typing import Optional, Any
import hashlib

class CacheService:
    """Redis-backed cache service"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True
        )

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None  # Fail gracefully

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = 3600
    ) -> bool:
        """Set value in cache with TTL"""
        try:
            serialized = json.dumps(value)
            await self.redis.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            return await self.redis.exists(key) > 0
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False

    async def get_ttl(self, key: str) -> int:
        """Get remaining TTL in seconds"""
        try:
            return await self.redis.ttl(key)
        except Exception as e:
            logger.error(f"Cache TTL error: {e}")
            return -1

    async def close(self):
        """Close Redis connection"""
        await self.redis.close()
```

### Cache Decorator

Make caching seamless with a decorator:

```python
import functools
from typing import Callable

def cached(
    ttl: int = 3600,
    key_prefix: str = ""
):
    """Decorator for caching function results"""

    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get cache service from first arg (self)
            cache_service = getattr(args[0], 'cache', None)

            if not cache_service:
                # No cache available, just call function
                return await func(*args, **kwargs)

            # Generate cache key
            cache_key = _generate_cache_key(
                key_prefix or func.__name__,
                args[1:],  # Skip self
                kwargs
            )

            # Try to get from cache
            cached_value = await cache_service.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return cached_value

            # Cache miss - call function
            logger.debug(f"Cache miss: {cache_key}")
            result = await func(*args, **kwargs)

            # Store in cache
            await cache_service.set(cache_key, result, ttl=ttl)

            return result

        return wrapper
    return decorator


def _generate_cache_key(prefix: str, args: tuple, kwargs: dict) -> str:
    """Generate deterministic cache key"""
    # Serialize args and kwargs
    key_parts = [prefix]

    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        else:
            # For complex objects, use hash
            key_parts.append(str(hash(str(arg))))

    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={v}")

    # Create hash of all parts for consistent key length
    key_string = ":".join(key_parts)
    key_hash = hashlib.md5(key_string.encode()).hexdigest()

    return f"{prefix}:{key_hash}"


# Usage
class FinancialAgent:
    def __init__(self, cache_service: CacheService):
        self.cache = cache_service

    @cached(ttl=3600, key_prefix="stock_analysis")
    async def analyze_stock(self, symbol: str, data: dict):
        """This method is now automatically cached!"""
        # Expensive LLM call
        return await self.llm.analyze(symbol, data)
```

---

## Cache Key Design

Good cache keys prevent collisions and make debugging easy.

### Key Structure

Use a hierarchical structure:

```
{namespace}:{entity}:{identifier}:{version}
```

**Examples**:
```
analysis:stock:AAPL:v1
analysis:stock:MSFT:v1
sentiment:news:abc123:v1
portfolio:user:user123:v2
```

### Key Design Patterns

**Pattern 1: Simple Symbol Cache**
```python
def get_cache_key(symbol: str) -> str:
    return f"analysis:stock:{symbol}"

# Keys: analysis:stock:AAPL, analysis:stock:MSFT
```

**Pattern 2: Include Data Hash**
```python
def get_cache_key(symbol: str, data: dict) -> str:
    # Hash the input data
    data_hash = hashlib.md5(
        json.dumps(data, sort_keys=True).encode()
    ).hexdigest()[:8]
    return f"analysis:stock:{symbol}:{data_hash}"

# Keys: analysis:stock:AAPL:a1b2c3d4, analysis:stock:AAPL:e5f6g7h8
# Different data = different cache key
```

**Pattern 3: Include Timestamp**
```python
def get_cache_key(symbol: str, data: dict) -> str:
    # Round to nearest hour
    hour_timestamp = int(time.time() // 3600)
    return f"analysis:stock:{symbol}:{hour_timestamp}"

# Keys auto-invalidate every hour
# analysis:stock:AAPL:478292, analysis:stock:AAPL:478293
```

**Pattern 4: Include Model Version**
```python
def get_cache_key(symbol: str, model: str = "claude-3-5") -> str:
    return f"analysis:stock:{symbol}:model:{model}"

# Different models = different cache
# analysis:stock:AAPL:model:claude-3-5
# analysis:stock:AAPL:model:groq-llama
```

### Best Practices

1. **Keep keys short** but descriptive
2. **Use consistent separators** (`:` recommended)
3. **Include version** to allow cache invalidation on schema changes
4. **Hash complex inputs** to avoid key length issues
5. **Namespace by type** to prevent collisions

---

## TTL Strategies

**TTL (Time To Live)** determines how long cached data remains valid. Choose wisely.

### TTL by Data Type

| Data Type | TTL | Reasoning |
|-----------|-----|-----------|
| Stock fundamentals | 1 day | Changes infrequently |
| News sentiment | 1 hour | News impact fades |
| Technical analysis | 15 minutes | Price-based, changes often |
| Earnings analysis | 1 week | Quarterly reports are stable |
| Market overview | 30 minutes | Market conditions shift |
| Historical analysis | 1 month | Past doesn't change |

### Dynamic TTL

Adjust TTL based on context:

```python
def get_ttl(symbol: str, analysis_type: str, market_hours: bool) -> int:
    """Smart TTL based on context"""

    base_ttls = {
        "fundamental": 86400,     # 1 day
        "sentiment": 3600,        # 1 hour
        "technical": 900,         # 15 minutes
        "earnings": 604800        # 1 week
    }

    ttl = base_ttls.get(analysis_type, 3600)

    # During market hours, use shorter TTL
    if market_hours and analysis_type in ["technical", "sentiment"]:
        ttl = ttl // 2

    # For volatile stocks, use shorter TTL
    if symbol in ["TSLA", "GME", "AMC"]:
        ttl = ttl // 2

    return ttl
```

### TTL Anti-Patterns

**Too Short**:
```python
# Bad: Cache expires too quickly, minimal benefit
ttl = 60  # 1 minute - why bother?
```

**Too Long**:
```python
# Bad: Stale data causes bad trading decisions
ttl = 604800  # 1 week for price-based analysis - dangerous!
```

**One Size Fits All**:
```python
# Bad: Everything cached for same duration
ttl = 3600  # Not all data has same freshness requirements
```

### Optimal TTL Strategy

```python
class SmartCache:
    """Cache with intelligent TTL"""

    TTL_CONFIG = {
        # (data_type, market_state) -> TTL
        ("fundamental", "open"): 3600,      # 1 hour
        ("fundamental", "closed"): 86400,   # 1 day
        ("sentiment", "open"): 900,         # 15 min
        ("sentiment", "closed"): 3600,      # 1 hour
        ("technical", "open"): 300,         # 5 min
        ("technical", "closed"): 1800,      # 30 min
    }

    def get_ttl(self, data_type: str) -> int:
        market_state = "open" if self.is_market_open() else "closed"
        key = (data_type, market_state)
        return self.TTL_CONFIG.get(key, 3600)  # Default 1 hour

    def is_market_open(self) -> bool:
        """Check if US markets are open"""
        from datetime import datetime, time
        import pytz

        now = datetime.now(pytz.timezone('America/New_York'))

        # Market open Monday-Friday, 9:30 AM - 4:00 PM ET
        if now.weekday() >= 5:  # Weekend
            return False

        market_open = time(9, 30)
        market_close = time(16, 0)

        return market_open <= now.time() <= market_close
```

---

## Cache Invalidation

> "There are only two hard things in Computer Science: cache invalidation and naming things." - Phil Karlton

### Invalidation Strategies

**Strategy 1: TTL-Based (Passive)**

Let cache expire naturally:
```python
# Set and forget
await cache.set("analysis:AAPL", data, ttl=3600)
# After 1 hour, cache expires automatically
```

**Pros**: Simple, no extra logic
**Cons**: May serve stale data until expiration

**Strategy 2: Event-Based (Active)**

Invalidate on specific events:
```python
async def on_earnings_release(symbol: str):
    """Invalidate cache when earnings released"""
    await cache.delete(f"analysis:stock:{symbol}")
    await cache.delete(f"fundamental:stock:{symbol}")
    logger.info(f"Invalidated cache for {symbol} after earnings")

async def on_major_news(symbol: str):
    """Invalidate on breaking news"""
    await cache.delete(f"sentiment:stock:{symbol}")
```

**Pros**: Fresh data when it matters
**Cons**: Requires event system

**Strategy 3: Version-Based**

Include version in key:
```python
CACHE_VERSION = "v2"  # Increment when schema changes

def get_cache_key(symbol: str) -> str:
    return f"analysis:stock:{symbol}:{CACHE_VERSION}"

# When you update analysis format:
CACHE_VERSION = "v3"  # Old cache keys ignored
```

**Pros**: Clean migration, no manual deletion
**Cons**: Wastes memory until old keys expire

**Strategy 4: Pattern-Based Deletion**

Delete multiple related keys:
```python
async def invalidate_stock(symbol: str):
    """Invalidate all cache entries for a stock"""
    pattern = f"*:stock:{symbol}*"

    # Get all matching keys
    keys = await cache.redis.keys(pattern)

    # Delete in batch
    if keys:
        await cache.redis.delete(*keys)
        logger.info(f"Invalidated {len(keys)} cache entries for {symbol}")
```

### Conditional Invalidation

Invalidate based on data changes:

```python
async def analyze_with_smart_cache(symbol: str, data: dict):
    """Cache only if data hasn't changed significantly"""

    cache_key = f"analysis:stock:{symbol}"

    # Get cached analysis and input data
    cached = await cache.get(cache_key)

    if cached:
        cached_data = cached.get("input_data")
        cached_result = cached.get("result")

        # Check if data changed significantly
        if not _data_changed_significantly(cached_data, data):
            logger.debug(f"Using cached result for {symbol}")
            return cached_result

    # Data changed or no cache - analyze
    result = await llm.analyze(symbol, data)

    # Cache with input data
    await cache.set(cache_key, {
        "input_data": data,
        "result": result
    }, ttl=3600)

    return result


def _data_changed_significantly(old: dict, new: dict) -> bool:
    """Check if data changed enough to invalidate cache"""

    thresholds = {
        "pe_ratio": 0.05,      # 5% change
        "revenue_growth": 0.02, # 2 percentage points
        "debt_ratio": 0.1       # 10% change
    }

    for key, threshold in thresholds.items():
        if key in old and key in new:
            old_val = old[key]
            new_val = new[key]

            # Calculate relative change
            if old_val != 0:
                change = abs(new_val - old_val) / abs(old_val)
                if change > threshold:
                    return True

    return False
```

---

## Cost Savings Analysis

Quantify the impact of caching with real numbers.

### Measuring Cache Performance

```python
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class CacheStats:
    """Track cache performance"""
    hits: int = 0
    misses: int = 0
    total_cost_without_cache: float = 0.0
    total_cost_with_cache: float = 0.0
    total_latency_without_cache: float = 0.0
    total_latency_with_cache: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def cost_savings(self) -> float:
        return self.total_cost_without_cache - self.total_cost_with_cache

    @property
    def savings_percentage(self) -> float:
        if self.total_cost_without_cache == 0:
            return 0.0
        return (self.cost_savings / self.total_cost_without_cache) * 100

    def report(self) -> str:
        return f"""
Cache Performance Report:
========================
Hit Rate: {self.hit_rate:.1%}
Hits: {self.hits}
Misses: {self.misses}

Cost Analysis:
-------------
Without Cache: ${self.total_cost_without_cache:.2f}
With Cache: ${self.total_cost_with_cache:.2f}
Savings: ${self.cost_savings:.2f} ({self.savings_percentage:.1f}%)

Latency Analysis:
----------------
Without Cache: {self.total_latency_without_cache:.0f}ms
With Cache: {self.total_latency_with_cache:.0f}ms
Speedup: {self.total_latency_without_cache / self.total_latency_with_cache if self.total_latency_with_cache > 0 else 0:.1f}x
"""


class CachedLLMClient:
    """LLM client with cost tracking"""

    def __init__(self, llm_client, cache_service):
        self.llm = llm_client
        self.cache = cache_service
        self.stats = CacheStats()

    async def analyze(self, symbol: str, data: dict) -> dict:
        """Analyze with caching and stat tracking"""
        cache_key = f"analysis:{symbol}"

        start_time = time.time()

        # Try cache
        cached = await self.cache.get(cache_key)

        if cached:
            # Cache hit
            latency = (time.time() - start_time) * 1000  # ms
            self.stats.hits += 1
            self.stats.total_latency_with_cache += latency
            # No API cost for cache hit
            return cached

        # Cache miss - call LLM
        result = await self.llm.analyze(symbol, data)

        latency = (time.time() - start_time) * 1000  # ms
        cost = self._estimate_cost(data, result)

        self.stats.misses += 1
        self.stats.total_cost_without_cache += cost
        self.stats.total_cost_with_cache += cost
        self.stats.total_latency_without_cache += latency
        self.stats.total_latency_with_cache += latency

        # Update cache
        await self.cache.set(cache_key, result, ttl=3600)

        return result

    def _estimate_cost(self, input_data: dict, output_data: dict) -> float:
        """Estimate API cost (example for Claude)"""
        # Rough estimate: $3 per 1M input tokens, $15 per 1M output tokens
        input_tokens = len(str(input_data)) * 0.25  # Rough approximation
        output_tokens = len(str(output_data)) * 0.25

        input_cost = (input_tokens / 1_000_000) * 3
        output_cost = (output_tokens / 1_000_000) * 15

        return input_cost + output_cost
```

### Example: One Day of Trading

```python
async def simulate_trading_day():
    """Simulate one day of trading with cache"""

    cache = CacheService()
    llm = ClaudeClient()
    client = CachedLLMClient(llm, cache)

    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"] * 10  # 50 stocks
    cycles = 26  # Every 15 min for 6.5 hours

    for cycle in range(cycles):
        for symbol in stocks:
            # Simulate stock data (changes slightly each cycle)
            data = generate_stock_data(symbol, cycle)
            await client.analyze(symbol, data)

        # Small delay between cycles
        await asyncio.sleep(1)

    # Print report
    print(client.stats.report())

# Expected output:
# Cache Performance Report:
# ========================
# Hit Rate: 92.3%
# Hits: 1200
# Misses: 100
#
# Cost Analysis:
# -------------
# Without Cache: $13.00
# With Cache: $1.00
# Savings: $12.00 (92.3%)
```

---

## Monitoring and Debugging

Track cache performance to optimize over time.

### Logging Cache Operations

```python
import logging

logger = logging.getLogger(__name__)

class MonitoredCache(CacheService):
    """Cache with detailed logging"""

    async def get(self, key: str):
        result = await super().get(key)

        if result:
            logger.info(f"CACHE_HIT: {key}")
        else:
            logger.info(f"CACHE_MISS: {key}")

        return result

    async def set(self, key: str, value: Any, ttl: int):
        result = await super().set(key, value, ttl)

        logger.info(f"CACHE_SET: {key} (TTL: {ttl}s)")

        return result
```

### Cache Metrics Dashboard

```python
async def get_cache_metrics() -> dict:
    """Get cache health metrics"""

    info = await cache.redis.info()
    stats = await cache.redis.info("stats")

    return {
        "used_memory": info.get("used_memory_human"),
        "connected_clients": info.get("connected_clients"),
        "total_keys": await cache.redis.dbsize(),
        "hit_rate": stats.get("keyspace_hits", 0) / (
            stats.get("keyspace_hits", 0) + stats.get("keyspace_misses", 1)
        ),
        "evicted_keys": stats.get("evicted_keys", 0),
        "expired_keys": stats.get("expired_keys", 0)
    }
```

### Debugging Cache Issues

**Issue 1: Low Hit Rate**

```python
# Check what keys exist
keys = await cache.redis.keys("analysis:*")
print(f"Found {len(keys)} cache entries")

# Check TTLs
for key in keys[:10]:
    ttl = await cache.get_ttl(key)
    print(f"{key}: {ttl}s remaining")
```

**Issue 2: Memory Usage**

```python
# Find largest keys
async def find_large_keys(pattern: str = "*", limit: int = 10):
    """Find keys using most memory"""
    keys = await cache.redis.keys(pattern)
    sizes = []

    for key in keys:
        size = await cache.redis.memory_usage(key)
        sizes.append((key, size))

    # Sort by size
    sizes.sort(key=lambda x: x[1], reverse=True)

    return sizes[:limit]

# Usage
large_keys = await find_large_keys()
for key, size in large_keys:
    print(f"{key}: {size / 1024:.1f} KB")
```

---

## Summary

### Quick Reference

```python
# Setup Redis cache
cache = CacheService("redis://localhost:6379")

# Basic operations
await cache.set("key", {"data": "value"}, ttl=3600)
result = await cache.get("key")
await cache.delete("key")

# Use decorator for automatic caching
class Agent:
    def __init__(self, cache):
        self.cache = cache

    @cached(ttl=3600, key_prefix="analysis")
    async def analyze(self, symbol, data):
        return await llm.analyze(symbol, data)
```

### Best Practices

1. **Always cache LLM responses** - saves money and improves latency
2. **Use appropriate TTLs** - balance freshness with cost savings
3. **Design good cache keys** - avoid collisions, enable debugging
4. **Implement invalidation** - don't serve stale data
5. **Monitor performance** - track hit rate and costs
6. **Fail gracefully** - cache errors shouldn't break your system
7. **Version your cache** - makes schema migrations easy

### Cost Impact

- **Without cache**: $300-400/month for moderate usage
- **With cache (80% hit rate)**: $60-80/month
- **With cache (95% hit rate)**: $15-20/month

**ROI**: Spend 1 hour setting up caching, save $300+/month forever.

### Next Steps

1. Set up Redis (Docker or local)
2. Implement `CacheService` class
3. Add caching to your `FinancialAgent`
4. Monitor hit rate and adjust TTLs
5. Read `llm_testing.md` for testing strategies

Caching is the lowest-hanging fruit for LLM cost optimization. Don't skip it!
