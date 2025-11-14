# Async Python: A Comprehensive Guide

## Table of Contents
1. [What is Async/Await?](#what-is-asyncawait)
2. [Synchronous vs Asynchronous](#synchronous-vs-asynchronous)
3. [When to Use Async](#when-to-use-async)
4. [Asyncio Basics](#asyncio-basics)
5. [Concurrent vs Parallel](#concurrent-vs-parallel)
6. [Common Pitfalls](#common-pitfalls)
7. [Best Practices](#best-practices)
8. [Real-World Examples](#real-world-examples)

---

## What is Async/Await?

Async/await is Python's way of writing **asynchronous** code that can handle multiple operations concurrently without blocking. Instead of waiting for slow operations (like network requests or database queries) to complete, your program can do other work while waiting.

### The Problem: Blocking I/O

Traditional synchronous code blocks execution while waiting for I/O operations:

```python
import time

def fetch_stock_price(symbol):
    """Simulate API call taking 2 seconds"""
    time.sleep(2)  # Blocks for 2 seconds
    return {"symbol": symbol, "price": 150.25}

# Fetch 3 stocks sequentially
start = time.time()
aapl = fetch_stock_price("AAPL")  # Wait 2 seconds
googl = fetch_stock_price("GOOGL")  # Wait 2 more seconds
msft = fetch_stock_price("MSFT")  # Wait 2 more seconds
end = time.time()

print(f"Total time: {end - start} seconds")  # ~6 seconds!
```

### The Solution: Async/Await

Async code can handle multiple operations concurrently:

```python
import asyncio

async def fetch_stock_price(symbol):
    """Async version - doesn't block"""
    await asyncio.sleep(2)  # Other tasks can run during this wait
    return {"symbol": symbol, "price": 150.25}

async def main():
    # Fetch 3 stocks concurrently
    start = time.time()
    results = await asyncio.gather(
        fetch_stock_price("AAPL"),
        fetch_stock_price("GOOGL"),
        fetch_stock_price("MSFT")
    )
    end = time.time()

    print(f"Total time: {end - start} seconds")  # ~2 seconds!

# Run the async function
asyncio.run(main())
```

### Key Concepts

- **`async def`**: Declares an asynchronous function (coroutine)
- **`await`**: Pauses execution until the awaited operation completes
- **Coroutine**: A function defined with `async def`
- **Event Loop**: Manages and schedules async tasks

---

## Synchronous vs Asynchronous

### Synchronous Code

Executes one operation at a time, waiting for each to complete:

```python
def process_orders():
    """Synchronous processing - one at a time"""
    order1 = submit_order("AAPL")  # Wait
    order2 = submit_order("GOOGL")  # Wait
    order3 = submit_order("MSFT")  # Wait
    return [order1, order2, order3]

# Total time = sum of all operations
```

**Analogy**: Like waiting in a single-file line at a store. Each customer must be fully served before the next can start.

### Asynchronous Code

Can juggle multiple operations, switching between them when they're waiting:

```python
async def process_orders():
    """Async processing - concurrent"""
    orders = await asyncio.gather(
        submit_order("AAPL"),   # Start all three
        submit_order("GOOGL"),  # at the same time
        submit_order("MSFT")
    )
    return orders

# Total time = longest single operation
```

**Analogy**: Like a chef preparing multiple dishes. While pasta is boiling, they can chop vegetables. Multiple tasks progress together.

### The Difference

| Synchronous | Asynchronous |
|------------|--------------|
| One task at a time | Multiple tasks concurrently |
| Blocks while waiting | Switches to other tasks while waiting |
| Simple, linear flow | More complex, event-driven |
| Total time = sum of operations | Total time = longest operation |
| Good for CPU-bound tasks | Good for I/O-bound tasks |

---

## When to Use Async

### Use Async For (I/O-Bound Operations)

**I/O-bound** means the program spends most time waiting for input/output:

```python
# ✅ GOOD use cases for async
async def good_async_examples():
    # Network requests (API calls)
    price = await fetch_from_api("https://api.example.com/price")

    # Database queries
    trades = await db.query("SELECT * FROM trades")

    # File I/O operations
    data = await read_file_async("large_data.csv")

    # WebSocket connections
    await websocket.send({"action": "subscribe"})

    # Any operation that waits for external resources
```

### Don't Use Async For (CPU-Bound Operations)

**CPU-bound** means the program spends most time doing calculations:

```python
# ❌ BAD use cases for async
async def bad_async_examples():
    # Heavy calculations - blocks the event loop
    result = await calculate_fibonacci(1000000)  # No benefit!

    # Data processing
    processed = await process_large_dataset(data)  # Still blocks!

    # Image/video processing
    compressed = await compress_video(video_file)  # Blocks!

    # Use multiprocessing/threading for these instead
```

### Decision Guide

```
Is your task I/O-bound (waiting for network/disk/database)?
├─ YES → Use async/await ✅
└─ NO → Is it CPU-bound (heavy calculations)?
    ├─ YES → Use multiprocessing/threading ⚙️
    └─ NO → Use regular synchronous code 📝
```

---

## Asyncio Basics

### Creating Async Functions

```python
# Define an async function
async def fetch_data(symbol):
    """This is a coroutine"""
    await asyncio.sleep(1)
    return f"Data for {symbol}"

# You CANNOT just call it like a normal function
# result = fetch_data("AAPL")  # This returns a coroutine object, not the result!

# You must await it inside another async function
async def main():
    result = await fetch_data("AAPL")  # ✅ Correct
    print(result)

# Or use asyncio.run() to start the event loop
asyncio.run(main())
```

### Running Multiple Tasks Concurrently

```python
import asyncio

async def fetch_price(symbol):
    await asyncio.sleep(1)  # Simulate API call
    return {symbol: 150.25}

async def main():
    # Method 1: asyncio.gather() - runs all tasks concurrently
    results = await asyncio.gather(
        fetch_price("AAPL"),
        fetch_price("GOOGL"),
        fetch_price("MSFT")
    )
    print(results)  # All three results

    # Method 2: asyncio.create_task() - more control
    task1 = asyncio.create_task(fetch_price("AAPL"))
    task2 = asyncio.create_task(fetch_price("GOOGL"))
    task3 = asyncio.create_task(fetch_price("MSFT"))

    # Wait for all tasks
    result1 = await task1
    result2 = await task2
    result3 = await task3

asyncio.run(main())
```

### Timeouts

```python
async def fetch_with_timeout(symbol):
    try:
        # Timeout after 5 seconds
        result = await asyncio.wait_for(
            fetch_price(symbol),
            timeout=5.0
        )
        return result
    except asyncio.TimeoutError:
        print(f"Timeout fetching {symbol}")
        return None
```

### Error Handling

```python
async def safe_fetch(symbol):
    try:
        result = await fetch_price(symbol)
        return result
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

async def main():
    # gather() with return_exceptions=True
    results = await asyncio.gather(
        fetch_price("AAPL"),
        fetch_price("INVALID"),
        fetch_price("GOOGL"),
        return_exceptions=True  # Don't stop on errors
    )

    # Results will include exception objects for failed tasks
    for result in results:
        if isinstance(result, Exception):
            print(f"Task failed: {result}")
        else:
            print(f"Task succeeded: {result}")
```

---

## Concurrent vs Parallel

This is a crucial distinction in async programming:

### Concurrent (Async)

**One thread, multiple tasks switching**. Tasks take turns, yielding control when waiting:

```python
async def concurrent_example():
    """
    Single thread handles multiple tasks.
    While one task waits (I/O), another can run.
    """
    tasks = [
        fetch_price("AAPL"),   # Start
        fetch_price("GOOGL"),  # Start
        fetch_price("MSFT")    # Start
    ]

    # All run on the same thread, switching during await points
    results = await asyncio.gather(*tasks)

# Timeline (simplified):
# 0.0s: Start AAPL fetch
# 0.0s: Start GOOGL fetch (while AAPL waits)
# 0.0s: Start MSFT fetch (while others wait)
# 1.0s: All complete (all ran concurrently on one thread)
```

### Parallel (Multiprocessing)

**Multiple threads/processes running simultaneously**. True simultaneous execution:

```python
from concurrent.futures import ProcessPoolExecutor

def cpu_intensive_task(n):
    """Heavy calculation - uses CPU"""
    return sum(i * i for i in range(n))

# Run on multiple CPU cores in parallel
with ProcessPoolExecutor() as executor:
    results = executor.map(cpu_intensive_task, [1000000, 2000000, 3000000])

# Timeline:
# Core 1: Working on 1000000
# Core 2: Working on 2000000  } All running at the
# Core 3: Working on 3000000  } exact same time
```

### Async vs Threading vs Multiprocessing

| Feature | Async | Threading | Multiprocessing |
|---------|-------|-----------|----------------|
| **Good for** | I/O-bound | I/O-bound | CPU-bound |
| **Execution** | Concurrent | Concurrent | Parallel |
| **Complexity** | Medium | High | High |
| **Overhead** | Low | Medium | High |
| **GIL Impact** | No | Yes | No |
| **Use in trading** | API calls | Database | Backtesting |

### The Global Interpreter Lock (GIL)

Python's GIL means only one thread executes Python code at a time. This makes threading unsuitable for CPU-bound tasks (use multiprocessing instead), but async works great for I/O-bound tasks.

---

## Common Pitfalls

### 1. Forgetting to Use `await`

```python
# WRONG - Doesn't actually wait for the result
async def wrong_example():
    result = fetch_price("AAPL")  # Returns coroutine, doesn't execute!
    print(result)  # Prints: <coroutine object fetch_price>

# RIGHT - Use await
async def correct_example():
    result = await fetch_price("AAPL")  # Actually executes and waits
    print(result)  # Prints: actual price data
```

### 2. Mixing Sync and Async Code

```python
# WRONG - Using sync library in async function
import requests  # Synchronous library

async def wrong_fetch(symbol):
    response = requests.get(f"https://api.example.com/{symbol}")
    # This BLOCKS the event loop! Defeats the purpose of async
    return response.json()

# RIGHT - Use async library
import httpx  # Asynchronous library

async def correct_fetch(symbol):
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.example.com/{symbol}")
        return response.json()
```

### 3. Not Running Async Code Properly

```python
# WRONG - Can't just call async function
result = fetch_price("AAPL")  # Returns coroutine object, doesn't run!

# RIGHT - Use asyncio.run() or await
asyncio.run(fetch_price("AAPL"))  # In scripts

# Or await in async context
async def main():
    result = await fetch_price("AAPL")  # In async functions
```

### 4. Creating Tasks Without Awaiting

```python
# WRONG - Task created but never executed
async def wrong_way():
    asyncio.create_task(fetch_price("AAPL"))
    # Function ends, task is garbage collected without running!

# RIGHT - Store and await tasks
async def correct_way():
    task = asyncio.create_task(fetch_price("AAPL"))
    result = await task
    return result
```

### 5. Blocking the Event Loop

```python
# WRONG - CPU-intensive operation blocks everything
async def wrong_calculation():
    result = expensive_calculation()  # Blocks for 10 seconds!
    return result

# RIGHT - Offload to thread/process
from concurrent.futures import ProcessPoolExecutor

async def correct_calculation():
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, expensive_calculation)
    return result
```

---

## Best Practices

### 1. Use Async Libraries

```python
# Use async-compatible libraries
import httpx      # Instead of requests
import aiofiles   # Instead of open()
import aiomysql   # Instead of mysql.connector
import asyncpg    # Instead of psycopg2
```

### 2. Set Timeouts

```python
async def fetch_with_timeout(symbol):
    """Always set timeouts to prevent hanging"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"/api/{symbol}")
            return response.json()
    except httpx.TimeoutException:
        return None
```

### 3. Handle Errors Gracefully

```python
async def safe_operation(symbol):
    """Wrap operations in try/except"""
    try:
        result = await fetch_price(symbol)
        return result
    except httpx.HTTPError as e:
        logger.error(f"HTTP error for {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error for {symbol}: {e}")
        return None
```

### 4. Use `asyncio.gather()` for Multiple Operations

```python
async def fetch_multiple(symbols):
    """Fetch multiple symbols concurrently"""
    tasks = [fetch_price(symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

### 5. Clean Up Resources

```python
async def proper_cleanup():
    """Use context managers for cleanup"""
    async with httpx.AsyncClient() as client:
        # Client is automatically closed
        response = await client.get("/api/data")
        return response.json()
```

---

## Real-World Examples

### Example 1: Fetching Multiple Stock Prices

```python
import asyncio
import httpx

async def fetch_stock_price(client, symbol):
    """Fetch price for a single stock"""
    response = await client.get(f"https://api.example.com/quote/{symbol}")
    data = response.json()
    return {symbol: data['price']}

async def fetch_portfolio_prices(symbols):
    """Fetch prices for entire portfolio concurrently"""
    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = [fetch_stock_price(client, symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        return results

# Usage
symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
prices = asyncio.run(fetch_portfolio_prices(symbols))
print(prices)
```

### Example 2: FastAPI Endpoint

```python
from fastapi import FastAPI
import httpx

app = FastAPI()

@app.get("/portfolio/{symbols}")
async def get_portfolio_value(symbols: str):
    """Calculate total portfolio value"""
    symbol_list = symbols.split(",")

    # Fetch all prices concurrently
    async with httpx.AsyncClient() as client:
        tasks = [
            client.get(f"https://api.example.com/quote/{s}")
            for s in symbol_list
        ]
        responses = await asyncio.gather(*tasks)

    # Calculate total value
    prices = [r.json()['price'] for r in responses]
    total = sum(prices)

    return {
        "symbols": symbol_list,
        "prices": prices,
        "total_value": total
    }
```

---

## Summary

Async/await is a powerful tool for handling I/O-bound operations efficiently. Key takeaways:

1. **Async is for I/O**: Use it for network, database, and file operations
2. **Concurrent, not parallel**: One thread handling multiple tasks
3. **Use `await`**: Always await async functions to get results
4. **Async libraries**: Use httpx, aiofiles, asyncpg, not sync equivalents
5. **Error handling**: Always handle exceptions in async code
6. **Timeouts**: Set timeouts to prevent hanging operations
7. **Gather for multiple**: Use `asyncio.gather()` for concurrent operations

In algorithmic trading, async programming allows you to fetch market data for hundreds of stocks simultaneously, execute multiple trades concurrently, and build high-performance trading systems.

---

## Next Steps

1. Read `fastapi_fundamentals.md` to see async in action with FastAPI
2. Complete the async exercises in `/exercises`
3. Practice converting sync code to async
4. Build an async API for fetching market data
5. Read `paper_trading.md` to integrate async with trading

Happy async coding!
