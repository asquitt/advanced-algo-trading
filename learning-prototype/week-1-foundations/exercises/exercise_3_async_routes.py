"""
Exercise 3: Async Routes

Objective: Understand async/await and concurrent request handling.

Time: 30 minutes
Difficulty: Medium

What you'll learn:
- Async functions
- Concurrent requests
- asyncio.sleep vs time.sleep
- Non-blocking I/O
"""

from fastapi import FastAPI
import asyncio
import time
from datetime import datetime
import uvicorn

app = FastAPI(title="Async Exercise")

# ============================================================================
# YOUR CODE HERE
# ============================================================================

# TODO #1: Create sync endpoint (blocking)
# This will block the entire server while sleeping!
# @app.get("/slow-sync")
# def slow_sync():
#     """BAD: Blocks the server for 2 seconds."""
#     time.sleep(2)  # Blocking!
#     return {"message": "Done (blocking)", "time": datetime.now().isoformat()}


# TODO #2: Create async endpoint (non-blocking)
# This will NOT block - other requests can process during sleep
# @app.get("/slow-async")
# async def slow_async():
#     """GOOD: Non-blocking sleep."""
#     await asyncio.sleep(2)  # Non-blocking!
#     return {"message": "Done (non-blocking)", "time": datetime.now().isoformat()}


# TODO #3: Create endpoint that simulates API call
# Simulate fetching data from external API (takes 1 second)
# @app.get("/fetch-data/{source}")
# async def fetch_data(source: str):
#     """Simulate fetching data from external source."""
#     start_time = time.time()
#
#     # Simulate API call
#     await asyncio.sleep(1)
#
#     elapsed = time.time() - start_time
#     return {
#         "source": source,
#         "data": f"Data from {source}",
#         "elapsed_seconds": round(elapsed, 2),
#         "timestamp": datetime.now().isoformat()
#     }


# TODO #4: Create endpoint that makes multiple concurrent API calls
# Fetch data from multiple sources at once (should take ~1 second total, not 3)
# @app.get("/fetch-multiple")
# async def fetch_multiple():
#     """Fetch from multiple sources concurrently."""
#     start_time = time.time()
#
#     # Run all fetches concurrently
#     results = await asyncio.gather(
#         fetch_data("source1"),
#         fetch_data("source2"),
#         fetch_data("source3")
#     )
#
#     elapsed = time.time() - start_time
#     return {
#         "results": results,
#         "total_elapsed": round(elapsed, 2),
#         "note": "All fetches ran concurrently!"
#     }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ============================================================================
# TESTING YOUR WORK
# ============================================================================

"""
Test concurrent vs sequential:

1. Start the server:
   python exercise_3_async_routes.py

2. Test blocking endpoint (open 3 terminals):
   Terminal 1: curl http://localhost:8000/slow-sync
   Terminal 2: curl http://localhost:8000/slow-sync
   Terminal 3: curl http://localhost:8000/slow-sync

   Total time: ~6 seconds (sequential)

3. Test non-blocking endpoint (open 3 terminals):
   Terminal 1: curl http://localhost:8000/slow-async
   Terminal 2: curl http://localhost:8000/slow-async
   Terminal 3: curl http://localhost:8000/slow-async

   Total time: ~2 seconds (concurrent)

4. Test multiple fetches:
   curl http://localhost:8000/fetch-multiple

   Should take ~1 second (not 3!) because all fetches run concurrently

5. Or use this Python script to test:

import requests
import time

def test_concurrent():
    start = time.time()

    # Make 3 requests concurrently (in reality, use asyncio or threads)
    # For demo, we'll just time sequential requests
    for i in range(3):
        response = requests.get("http://localhost:8000/slow-async")
        print(f"Response {i+1}: {response.json()}")

    elapsed = time.time() - start
    print(f"Total time: {elapsed:.2f} seconds")
    # With async server, this will be much faster!

test_concurrent()
"""


# ============================================================================
# BONUS CHALLENGES
# ============================================================================

"""
If you finish early, try these:

1. Add timeout handling:
   @app.get("/with-timeout")
   async def with_timeout():
       try:
           result = await asyncio.wait_for(
               slow_operation(),
               timeout=1.0  # 1 second timeout
           )
           return {"result": result}
       except asyncio.TimeoutError:
           raise HTTPException(status_code=504, detail="Operation timed out")

2. Add rate limiting:
   from fastapi import HTTPException
   import asyncio

   request_times = []

   @app.get("/rate-limited")
   async def rate_limited():
       now = time.time()
       # Remove old requests (older than 60 seconds)
       request_times[:] = [t for t in request_times if now - t < 60]

       if len(request_times) >= 10:  # Max 10 requests per minute
           raise HTTPException(status_code=429, detail="Rate limit exceeded")

       request_times.append(now)
       return {"message": "OK", "requests_in_last_minute": len(request_times)}

3. Add background tasks:
   from fastapi import BackgroundTasks

   def write_log(message: str):
       time.sleep(2)  # Simulate slow logging
       print(f"LOG: {message}")

   @app.post("/process")
   async def process(background_tasks: BackgroundTasks):
       background_tasks.add_task(write_log, "Processing started")
       return {"message": "Processing in background"}
"""
