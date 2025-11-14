# Week 1: Foundations - Building Your First Trading API

**Time**: 10-12 hours | **Difficulty**: Beginner | **Prerequisites**: Python 3.11+, basic REST API knowledge

---

## 🎯 Learning Goals

By the end of this week, you will:

1. ✅ Build a working REST API with FastAPI
2. ✅ Understand Pydantic models for data validation
3. ✅ Integrate with Alpaca paper trading API
4. ✅ Execute your first paper trade programmatically
5. ✅ Write basic tests with pytest
6. ✅ Configure environment variables securely
7. ✅ Handle errors gracefully
8. ✅ Understand async/await in Python

---

## 📚 What You'll Build

A **Simple Trading API** that can:
- Generate basic BUY/SELL/HOLD signals
- Execute paper trades via Alpaca
- Track portfolio value
- Return JSON responses with validation
- Handle errors without crashing

**Example Usage**:
```bash
# Start your API
python starter-code/main.py

# Generate signal
curl http://localhost:8000/signal/AAPL

# Execute trade
curl -X POST http://localhost:8000/trade -d '{"symbol":"AAPL","side":"buy","quantity":10}'

# Check portfolio
curl http://localhost:8000/portfolio
```

---

## 🗺️ Learning Path

### Day 1-2: FastAPI Fundamentals (3-4 hours)
1. Read `notes/fastapi_fundamentals.md`
2. Complete `starter-code/main.py` TODOs
3. Run and test your API
4. Do Exercise 1 & 2

**Key Concepts**:
- Route decorators (`@app.get`, `@app.post`)
- Path parameters (`/signal/{symbol}`)
- Query parameters (`?confidence=0.8`)
- Request/Response models
- Async functions

### Day 2-3: Pydantic Models (2-3 hours)
1. Read `notes/pydantic_explained.md`
2. Complete `starter-code/models.py` TODOs
3. Run validation tests
4. Do Exercise 3

**Key Concepts**:
- Field validation
- Type hints
- Default values
- Validators
- Model serialization

### Day 3-4: Paper Trading Integration (3-4 hours)
1. Read `notes/paper_trading.md`
2. Get Alpaca API keys (free)
3. Complete `starter-code/broker.py` TODOs
4. Execute your first paper trade!
5. Do Exercise 4

**Key Concepts**:
- REST API clients
- Authentication (API keys)
- Order types (market, limit)
- Position tracking
- Error handling

### Day 4-5: Testing & Polish (2-3 hours)
1. Read `notes/testing_basics.md`
2. Write tests for your code
3. Run `scripts/test_week1.sh`
4. Do Exercise 5
5. Complete final validation

**Key Concepts**:
- pytest basics
- Mocking external APIs
- Test fixtures
- Assertions
- Code coverage

---

## 📋 Starter Code Overview

### main.py (FastAPI Application)
```python
"""
TODO #1: Create FastAPI app instance
TODO #2: Add CORS middleware
TODO #3: Implement /signal/{symbol} endpoint
TODO #4: Implement /trade endpoint
TODO #5: Implement /portfolio endpoint
TODO #6: Add error handling
TODO #7: Run with uvicorn
"""
```

**What You'll Learn**:
- How to structure a FastAPI application
- Routing and endpoint design
- Middleware configuration
- Error handling patterns

### models.py (Pydantic Models)
```python
"""
TODO #1: Create TradingSignal model
TODO #2: Create TradeRequest model
TODO #3: Create TradeResponse model
TODO #4: Add field validators
TODO #5: Add example values for docs
"""
```

**What You'll Learn**:
- Type safety in Python
- Automatic validation
- JSON serialization
- Documentation generation

### broker.py (Alpaca Integration)
```python
"""
TODO #1: Initialize Alpaca client
TODO #2: Implement get_account()
TODO #3: Implement get_positions()
TODO #4: Implement execute_trade()
TODO #5: Add error handling
"""
```

**What You'll Learn**:
- External API integration
- Async/await patterns
- API authentication
- Response parsing

### config.py (Configuration)
```python
"""
TODO #1: Create Settings class
TODO #2: Add API key fields
TODO #3: Add validation
TODO #4: Load from .env file
"""
```

**What You'll Learn**:
- Environment variables
- Configuration management
- Security best practices
- Pydantic Settings

---

## 🎓 Exercises

### Exercise 1: FastAPI Hello World
**File**: `exercises/exercise_1_fastapi_basics.py`

**Objective**: Create a simple API with 3 endpoints

```python
# TODO: Create endpoints for:
# 1. GET / - Return welcome message
# 2. GET /health - Return {"status": "healthy"}
# 3. GET /echo/{message} - Echo back the message
```

**Test**:
```bash
python exercise_1_fastapi_basics.py
# Then visit http://localhost:8000/docs
```

**Success Criteria**:
- All 3 endpoints work
- Swagger UI shows documentation
- Returns proper JSON

---

### Exercise 2: Pydantic Validation
**File**: `exercises/exercise_2_pydantic_models.py`

**Objective**: Create validated data models

```python
# TODO: Create models for:
# 1. Stock (symbol, price, volume)
# 2. Portfolio (stocks: List[Stock], total_value)
# 3. Add validators to ensure price > 0
```

**Test**:
```python
# Should work
stock = Stock(symbol="AAPL", price=150.0, volume=1000000)

# Should fail validation
stock = Stock(symbol="AAPL", price=-10.0, volume=1000000)  # Error!
```

**Success Criteria**:
- Models validate correctly
- Invalid data raises errors
- Can serialize to JSON

---

### Exercise 3: Async Routes
**File**: `exercises/exercise_3_async_routes.py`

**Objective**: Understand async/await

```python
# TODO: Create async endpoint that:
# 1. Sleeps for 2 seconds (simulating API call)
# 2. Returns current time
# 3. Can handle multiple concurrent requests
```

**Test**:
```bash
# These should complete in ~2 seconds total (concurrent), not 6 seconds
curl http://localhost:8000/slow &
curl http://localhost:8000/slow &
curl http://localhost:8000/slow &
```

**Success Criteria**:
- Requests run concurrently
- No blocking
- Proper async/await usage

---

### Exercise 4: Error Handling
**File**: `exercises/exercise_4_error_handling.py`

**Objective**: Handle errors gracefully

```python
# TODO: Create endpoint that:
# 1. Fetches stock price (can fail)
# 2. Returns proper HTTP status codes
# 3. Returns meaningful error messages
# 4. Logs errors
```

**Test**:
```bash
# Valid request - should return 200
curl http://localhost:8000/price/AAPL

# Invalid request - should return 404 with message
curl http://localhost:8000/price/INVALID

# Server error - should return 500 with safe message
curl http://localhost:8000/price/ERROR
```

**Success Criteria**:
- Proper HTTP status codes
- User-friendly error messages
- Errors logged but not exposed
- No crashes

---

### Exercise 5: Integration Test
**File**: `exercises/exercise_5_integration.py`

**Objective**: Test complete workflow

```python
# TODO: Write tests that:
# 1. Generate a signal for AAPL
# 2. Execute a trade based on signal
# 3. Verify portfolio updated
# 4. Use TestClient for testing
```

**Test**:
```bash
pytest exercises/exercise_5_integration.py -v
```

**Success Criteria**:
- All tests pass
- Workflow works end-to-end
- Proper mocking of Alpaca API
- Good test coverage

---

## 📖 Detailed Notes

All notes are in the `notes/` folder:

1. **fastapi_fundamentals.md** (2,000 words)
   - FastAPI introduction
   - Routing and endpoints
   - Request/Response models
   - Async programming
   - Middleware
   - Code examples

2. **pydantic_explained.md** (1,500 words)
   - Why Pydantic?
   - Type hints and validation
   - Field types
   - Validators and root validators
   - Model Config
   - Serialization

3. **async_python.md** (1,200 words)
   - What is async/await?
   - When to use async
   - Common pitfalls
   - asyncio basics
   - Concurrent vs parallel

4. **paper_trading.md** (1,800 words)
   - What is paper trading?
   - Alpaca API overview
   - Authentication
   - Order types
   - Position tracking
   - Risk management basics

5. **testing_basics.md** (1,500 words)
   - pytest introduction
   - Writing good tests
   - Fixtures
   - Mocking
   - Test coverage
   - TDD basics

---

## 🛠️ Scripts

### test_week1.sh
Runs all your tests to verify completion.

```bash
cd scripts
./test_week1.sh
```

**What it does**:
1. Checks code syntax
2. Runs all exercises
3. Validates starter code TODOs completed
4. Runs integration tests
5. Checks code coverage
6. Provides detailed feedback

### run_api.sh
Starts your API for testing.

```bash
cd scripts
./run_api.sh
```

**What it does**:
1. Activates virtual environment
2. Sets environment variables
3. Starts uvicorn server
4. Opens browser to Swagger UI

### validate.py
Final validation before moving to Week 2.

```bash
cd scripts
python validate.py
```

**Checks**:
- All TODOs completed
- All exercises passing
- Tests passing
- Code quality
- Documentation

---

## ✅ Completion Checklist

Before moving to Week 2, ensure:

- [ ] Read all notes in `/notes`
- [ ] Completed all TODOs in starter code
- [ ] All 5 exercises passing
- [ ] `test_week1.sh` passes
- [ ] `validate.py` passes
- [ ] Can explain:
  - [ ] How FastAPI routing works
  - [ ] What Pydantic does
  - [ ] Difference between sync and async
  - [ ] How to execute a paper trade
  - [ ] Basic pytest usage

**Time Spent**: _____ hours (target: 10-12)

---

## 🎯 Success Metrics

You've mastered Week 1 if you can:

1. **Build APIs**: Create a FastAPI endpoint from scratch in <5 minutes
2. **Validate Data**: Write Pydantic models with proper validation
3. **Trade**: Execute a paper trade via Alpaca without looking at docs
4. **Test**: Write a pytest test for any function
5. **Debug**: Read error messages and fix issues independently

---

## 💡 Common Mistakes & Solutions

### Mistake 1: Forgetting async/await
**Problem**: Route is slow, blocking

**Solution**:
```python
# Wrong
@app.get("/data")
def get_data():
    result = slow_api_call()  # Blocks!
    return result

# Right
@app.get("/data")
async def get_data():
    result = await slow_api_call_async()  # Non-blocking
    return result
```

### Mistake 2: Not validating input
**Problem**: Bad data crashes server

**Solution**:
```python
# Use Pydantic models for all inputs
@app.post("/trade")
async def trade(request: TradeRequest):  # Validates automatically!
    ...
```

### Mistake 3: Exposing API keys
**Problem**: Keys in code

**Solution**:
```python
# Wrong
API_KEY = "pk_12345"  # Don't do this!

# Right
from config import settings
API_KEY = settings.alpaca_api_key  # From .env file
```

---

## 📚 Additional Resources

### Must-Read
- FastAPI Tutorial: https://fastapi.tiangolo.com/tutorial/
- Pydantic Docs: https://docs.pydantic.dev/latest/
- Alpaca Docs: https://alpaca.markets/docs/

### Recommended
- "FastAPI Best Practices" (GitHub repo)
- "Async Python" by Caleb Hattingh
- Real Python FastAPI articles

### Video Tutorials
- "FastAPI - Full Course" (freeCodeCamp)
- "Async Python for Web Developers"
- "Paper Trading Tutorial"

---

## 🏆 Bonus Challenges

If you finish early, try these:

1. **Add WebSocket Support**: Stream live price updates
2. **Implement Caching**: Use Redis to cache API responses
3. **Add Authentication**: Require API key for endpoints
4. **Deploy to Cloud**: Host your API on Heroku/Railway
5. **Build a CLI**: Create a command-line tool to interact with your API

---

## 🚀 Next Week Preview

**Week 2: LLM Integration**

You'll learn to:
- Integrate Groq and Claude APIs
- Write prompts for financial analysis
- Parse LLM responses into structured data
- Optimize costs with caching
- Build AI agents for trading

Get excited! Week 2 is where the magic happens! 🎩✨

---

**Ready? Start with `notes/fastapi_fundamentals.md` and then dive into `starter-code/main.py`!**

Good luck! You got this! 💪
