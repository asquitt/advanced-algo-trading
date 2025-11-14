# FastAPI Fundamentals

## Table of Contents
1. [Introduction to FastAPI](#introduction-to-fastapi)
2. [Why FastAPI?](#why-fastapi)
3. [Setting Up Your First API](#setting-up-your-first-api)
4. [Routing and Endpoints](#routing-and-endpoints)
5. [Request and Response Models](#request-and-response-models)
6. [Async Programming in FastAPI](#async-programming-in-fastapi)
7. [Middleware and CORS](#middleware-and-cors)
8. [Error Handling](#error-handling)
9. [Best Practices](#best-practices)
10. [Common Mistakes](#common-mistakes)

---

## Introduction to FastAPI

FastAPI is a modern, high-performance web framework for building APIs with Python 3.7+ based on standard Python type hints. Created by Sebastián Ramírez, it has quickly become one of the most popular choices for building production-ready APIs in Python.

The framework is built on top of **Starlette** (for web handling) and **Pydantic** (for data validation), combining their strengths to provide an exceptional developer experience. FastAPI is designed to be easy to use, fast to code, and production-ready with automatic interactive documentation.

### Key Features

- **Fast Performance**: Comparable to NodeJS and Go, thanks to Starlette and Pydantic
- **Type Safety**: Full type hints support with automatic validation
- **Auto Documentation**: Swagger UI and ReDoc generated automatically
- **Async Support**: Native async/await support for concurrent operations
- **Data Validation**: Automatic request/response validation via Pydantic
- **Modern Python**: Uses Python 3.6+ features like type hints and async

---

## Why FastAPI?

### Performance

FastAPI is one of the fastest Python frameworks available, with performance comparable to NodeJS and Go frameworks. This is crucial for algorithmic trading where latency matters.

```python
# FastAPI can handle thousands of concurrent requests efficiently
@app.get("/price/{symbol}")
async def get_price(symbol: str):
    # Non-blocking I/O operations
    price = await fetch_price_from_api(symbol)
    return {"symbol": symbol, "price": price}
```

### Developer Experience

The framework prioritizes developer productivity:

- **Auto-completion**: Full IDE support with type hints
- **Less code**: Minimal boilerplate compared to Flask or Django
- **Clear errors**: Helpful validation errors out of the box
- **Interactive docs**: Test endpoints directly in the browser

### Production-Ready

FastAPI includes everything you need for production:

- Built-in security utilities (OAuth2, JWT)
- CORS middleware configuration
- Background tasks support
- WebSocket support
- Dependency injection system

---

## Setting Up Your First API

### Installation

```bash
# Install FastAPI and ASGI server
pip install fastapi uvicorn[standard]

# Optional: Install for development
pip install python-multipart  # For form data
pip install httpx  # For async HTTP client
```

### Hello World Example

```python
# main.py
from fastapi import FastAPI

app = FastAPI(
    title="Trading API",
    description="A simple algorithmic trading API",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "Welcome to Trading API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Run with: uvicorn main:app --reload
```

### Running the Server

```bash
# Development mode with auto-reload
uvicorn main:app --reload --port 8000

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

Now visit:
- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

---

## Routing and Endpoints

### Path Parameters

Path parameters are variables in the URL path that are extracted and passed to your function.

```python
@app.get("/stock/{symbol}")
async def get_stock(symbol: str):
    """Get stock information by symbol"""
    return {
        "symbol": symbol.upper(),
        "price": 150.25,
        "exchange": "NASDAQ"
    }

# Usage: GET /stock/AAPL
# Returns: {"symbol": "AAPL", "price": 150.25, "exchange": "NASDAQ"}
```

### Query Parameters

Query parameters are optional parameters that come after `?` in the URL.

```python
from typing import Optional

@app.get("/search")
async def search_stocks(
    query: str,
    limit: int = 10,
    exchange: Optional[str] = None
):
    """Search for stocks with optional filters"""
    return {
        "query": query,
        "limit": limit,
        "exchange": exchange,
        "results": []
    }

# Usage: GET /search?query=tech&limit=5&exchange=NASDAQ
```

### Request Body (POST/PUT)

For creating or updating resources, use request bodies with Pydantic models.

```python
from pydantic import BaseModel

class TradeOrder(BaseModel):
    symbol: str
    quantity: int
    side: str  # "buy" or "sell"
    order_type: str = "market"

@app.post("/orders")
async def create_order(order: TradeOrder):
    """Execute a trade order"""
    return {
        "status": "submitted",
        "order": order.dict(),
        "order_id": "12345"
    }
```

### Multiple HTTP Methods

```python
@app.get("/orders/{order_id}")
async def get_order(order_id: str):
    """Retrieve an order"""
    return {"order_id": order_id, "status": "filled"}

@app.put("/orders/{order_id}")
async def update_order(order_id: str, order: TradeOrder):
    """Update an existing order"""
    return {"order_id": order_id, "updated": True}

@app.delete("/orders/{order_id}")
async def cancel_order(order_id: str):
    """Cancel an order"""
    return {"order_id": order_id, "cancelled": True}
```

---

## Request and Response Models

### Defining Models

Pydantic models provide automatic validation and serialization.

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class StockPrice(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=5)
    price: float = Field(..., gt=0)
    volume: int = Field(..., ge=0)
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "price": 150.25,
                "volume": 1000000,
                "timestamp": "2025-11-14T10:30:00"
            }
        }

class Portfolio(BaseModel):
    account_id: str
    positions: List[StockPrice]
    total_value: float
    cash_available: float
```

### Using Response Models

Response models ensure consistent API responses and automatic documentation.

```python
@app.get("/portfolio", response_model=Portfolio)
async def get_portfolio():
    """Get current portfolio"""
    return {
        "account_id": "ACC123",
        "positions": [
            {"symbol": "AAPL", "price": 150.25, "volume": 100},
            {"symbol": "GOOGL", "price": 2800.50, "volume": 50}
        ],
        "total_value": 155025.00,
        "cash_available": 10000.00
    }
```

### Status Codes

```python
from fastapi import HTTPException, status

@app.post("/orders", status_code=status.HTTP_201_CREATED)
async def create_order(order: TradeOrder):
    """Create a new order"""
    # Validation logic
    if order.quantity <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Quantity must be positive"
        )

    return {"order_id": "12345", "status": "created"}

@app.get("/orders/{order_id}")
async def get_order(order_id: str):
    """Get order by ID"""
    # Check if order exists
    order = find_order(order_id)
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found"
        )

    return order
```

---

## Async Programming in FastAPI

### When to Use Async

Use `async def` for I/O-bound operations like:
- Database queries
- External API calls
- File I/O operations
- Network requests

Use regular `def` for CPU-bound operations:
- Heavy calculations
- Data processing
- Image/video processing

### Async Example

```python
import httpx

# Async function for external API calls
@app.get("/market-data/{symbol}")
async def get_market_data(symbol: str):
    """Fetch real-time market data"""
    async with httpx.AsyncClient() as client:
        # Non-blocking HTTP request
        response = await client.get(
            f"https://api.example.com/quote/{symbol}"
        )
        data = response.json()

    return {
        "symbol": symbol,
        "data": data,
        "fetched_at": datetime.now()
    }
```

### Concurrent Operations

```python
import asyncio

@app.get("/multiple-quotes")
async def get_multiple_quotes(symbols: str):
    """Fetch multiple stock quotes concurrently"""
    symbol_list = symbols.split(",")

    # Run multiple API calls concurrently
    tasks = [fetch_quote(symbol) for symbol in symbol_list]
    results = await asyncio.gather(*tasks)

    return {"quotes": results}

async def fetch_quote(symbol: str):
    """Helper function to fetch a single quote"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.example.com/quote/{symbol}"
        )
        return response.json()
```

### Background Tasks

```python
from fastapi import BackgroundTasks

def send_notification(email: str, message: str):
    """Send email notification"""
    print(f"Sending email to {email}: {message}")
    # Email sending logic here

@app.post("/orders")
async def create_order(
    order: TradeOrder,
    background_tasks: BackgroundTasks
):
    """Create order and send notification in background"""
    # Create order
    order_id = save_order(order)

    # Schedule background task
    background_tasks.add_task(
        send_notification,
        "trader@example.com",
        f"Order {order_id} created"
    )

    return {"order_id": order_id, "status": "submitted"}
```

---

## Middleware and CORS

### CORS Middleware

CORS (Cross-Origin Resource Sharing) is essential for allowing frontend applications to access your API.

```python
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "https://yourdomain.com"   # Production frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# For development (less secure)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Custom Middleware

```python
from fastapi import Request
import time

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests"""
    print(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    print(f"Response status: {response.status_code}")
    return response
```

---

## Error Handling

### Global Exception Handler

```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError exceptions globally"""
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid value", "detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred"
        }
    )
```

### Custom Exceptions

```python
class TradingException(Exception):
    """Base exception for trading operations"""
    pass

class InsufficientFundsException(TradingException):
    """Raised when account has insufficient funds"""
    pass

class InvalidSymbolException(TradingException):
    """Raised when stock symbol is invalid"""
    pass

@app.exception_handler(InsufficientFundsException)
async def insufficient_funds_handler(request: Request, exc: InsufficientFundsException):
    return JSONResponse(
        status_code=402,  # Payment Required
        content={"error": "Insufficient funds", "detail": str(exc)}
    )

@app.post("/orders")
async def create_order(order: TradeOrder):
    """Create order with custom error handling"""
    if not validate_symbol(order.symbol):
        raise InvalidSymbolException(f"Invalid symbol: {order.symbol}")

    if not has_sufficient_funds(order):
        raise InsufficientFundsException("Not enough cash for this trade")

    return {"order_id": "12345"}
```

---

## Best Practices

### 1. Use Type Hints Everywhere

```python
from typing import List, Optional, Dict, Any

@app.get("/stocks")
async def get_stocks(
    exchange: Optional[str] = None,
    limit: int = 100
) -> Dict[str, Any]:
    """Type hints improve code quality and IDE support"""
    stocks: List[Dict[str, Any]] = fetch_stocks(exchange, limit)
    return {"count": len(stocks), "stocks": stocks}
```

### 2. Organize Code by Routers

```python
# routers/trading.py
from fastapi import APIRouter

router = APIRouter(prefix="/trading", tags=["trading"])

@router.post("/orders")
async def create_order(order: TradeOrder):
    return {"status": "created"}

# main.py
from routers import trading

app.include_router(trading.router)
```

### 3. Use Dependency Injection

```python
from fastapi import Depends

def get_api_key(api_key: str = Header(None)):
    """Validate API key"""
    if api_key != "secret-key":
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

@app.get("/protected")
async def protected_route(api_key: str = Depends(get_api_key)):
    """This route requires a valid API key"""
    return {"message": "Access granted"}
```

### 4. Environment Configuration

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Trading API"
    api_key: str
    database_url: str

    class Config:
        env_file = ".env"

settings = Settings()

@app.get("/config")
async def get_config():
    return {"app_name": settings.app_name}
```

### 5. Use Response Models

```python
# Always define response models for consistent APIs
@app.get("/portfolio", response_model=Portfolio)
async def get_portfolio():
    # Only fields defined in Portfolio will be returned
    return get_portfolio_data()
```

---

## Common Mistakes

### 1. Mixing Sync and Async Incorrectly

```python
# WRONG - Blocking async function
@app.get("/data")
async def get_data():
    result = requests.get("https://api.example.com")  # Blocking!
    return result.json()

# RIGHT - Use async HTTP client
@app.get("/data")
async def get_data():
    async with httpx.AsyncClient() as client:
        result = await client.get("https://api.example.com")
        return result.json()
```

### 2. Not Using Response Models

```python
# WRONG - Returns raw dict, no validation
@app.get("/user")
async def get_user():
    return {"name": "John", "age": "thirty"}  # Age should be int!

# RIGHT - Response model validates output
@app.get("/user", response_model=User)
async def get_user():
    return {"name": "John", "age": 30}
```

### 3. Ignoring Error Handling

```python
# WRONG - Unhandled exceptions crash the API
@app.get("/stock/{symbol}")
async def get_stock(symbol: str):
    return fetch_stock(symbol)  # What if this fails?

# RIGHT - Proper error handling
@app.get("/stock/{symbol}")
async def get_stock(symbol: str):
    try:
        stock = fetch_stock(symbol)
        if not stock:
            raise HTTPException(404, "Stock not found")
        return stock
    except Exception as e:
        raise HTTPException(500, f"Server error: {str(e)}")
```

### 4. Exposing Sensitive Information

```python
# WRONG - Exposing internal errors
@app.get("/data")
async def get_data():
    try:
        return fetch_data()
    except Exception as e:
        return {"error": str(e)}  # May expose system info!

# RIGHT - Safe error messages
@app.get("/data")
async def get_data():
    try:
        return fetch_data()
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise HTTPException(500, "Unable to fetch data")
```

### 5. Not Using Path Operation Configuration

```python
# WRONG - No metadata
@app.get("/stocks")
async def get_stocks():
    return []

# RIGHT - Proper documentation
@app.get(
    "/stocks",
    summary="List all stocks",
    description="Retrieve a list of all available stocks with filters",
    response_description="List of stock objects",
    tags=["stocks"]
)
async def get_stocks():
    """
    Get all stocks from the database.

    - **exchange**: Filter by exchange (optional)
    - **limit**: Maximum number of results
    """
    return []
```

---

## Summary

FastAPI is a powerful, modern framework that makes building APIs in Python a joy. Key takeaways:

1. **Type hints** are your friend - use them everywhere
2. **Async/await** for I/O operations, regular functions for CPU-bound tasks
3. **Pydantic models** for all request/response data
4. **Error handling** should be explicit and user-friendly
5. **Documentation** is automatic - just write good code
6. **CORS** must be configured for frontend access
7. **Dependency injection** keeps code clean and testable

FastAPI's combination of performance, developer experience, and production-readiness makes it ideal for building algorithmic trading systems. With automatic validation, serialization, and documentation, you can focus on your trading logic rather than boilerplate code.

---

## Next Steps

1. Read `pydantic_explained.md` to master data validation
2. Study `async_python.md` for deeper async understanding
3. Complete the FastAPI exercises in `/exercises`
4. Build your first trading API endpoint
5. Test everything with the interactive docs at `/docs`

Happy coding!
