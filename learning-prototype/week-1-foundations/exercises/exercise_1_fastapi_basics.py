"""
Exercise 1: FastAPI Basics

Objective: Create a simple API with 3 endpoints to learn FastAPI fundamentals.

Time: 30 minutes
Difficulty: Easy

What you'll learn:
- Creating FastAPI app
- Defining routes
- Path parameters
- Returning JSON responses
"""

from fastapi import FastAPI
import uvicorn

# ============================================================================
# YOUR CODE HERE
# ============================================================================

# TODO #1: Create FastAPI app instance
# HINT: app = FastAPI(title="Exercise 1 API", version="1.0.0")
app = None  # Replace with your code


# TODO #2: Create GET / endpoint
# Should return: {"message": "Welcome to my API!", "version": "1.0.0"}
# HINT:
# @app.get("/")
# async def root():
#     return {"message": "Welcome to my API!", "version": "1.0.0"}


# TODO #3: Create GET /health endpoint
# Should return: {"status": "healthy"}
# HINT:
# @app.get("/health")
# async def health():
#     return {"status": "healthy"}


# TODO #4: Create GET /echo/{message} endpoint
# Should return: {"original": message, "uppercase": message.upper(), "length": len(message)}
# HINT:
# @app.get("/echo/{message}")
# async def echo(message: str):
#     return {
#         "original": message,
#         "uppercase": message.upper(),
#         "length": len(message)
#     }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # TODO #5: Run the app with uvicorn
    # HINT: uvicorn.run(app, host="0.0.0.0", port=8000)
    pass


# ============================================================================
# TESTING YOUR WORK
# ============================================================================

"""
Once completed, test your API:

1. Run the file:
   python exercise_1_fastapi_basics.py

2. Open browser to http://localhost:8000/docs
   You should see automatic Swagger documentation!

3. Test each endpoint:
   - GET http://localhost:8000/
   - GET http://localhost:8000/health
   - GET http://localhost:8000/echo/hello

4. Or use curl:
   curl http://localhost:8000/
   curl http://localhost:8000/health
   curl http://localhost:8000/echo/FastAPI

Expected outputs:
- / → {"message": "Welcome to my API!", "version": "1.0.0"}
- /health → {"status": "healthy"}
- /echo/FastAPI → {"original": "FastAPI", "uppercase": "FASTAPI", "length": 7}
"""


# ============================================================================
# BONUS CHALLENGES
# ============================================================================

"""
If you finish early, try these:

1. Add query parameters:
   @app.get("/greet")
   async def greet(name: str = "World"):
       return {"message": f"Hello, {name}!"}

   Test: http://localhost:8000/greet?name=Alice

2. Add POST endpoint:
   from pydantic import BaseModel

   class Message(BaseModel):
       text: str

   @app.post("/reverse")
   async def reverse(msg: Message):
       return {"reversed": msg.text[::-1]}

3. Add error handling:
   @app.get("/divide/{a}/{b}")
   async def divide(a: int, b: int):
       if b == 0:
           raise HTTPException(status_code=400, detail="Cannot divide by zero")
       return {"result": a / b}
"""
